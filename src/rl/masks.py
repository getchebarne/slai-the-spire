import warnings

import numpy as np
import slai
import torch
from slai import ActionType
from slai import Card

from src.rl.encoding.card import SLICE_KINDS_CARD
from src.rl.encoding.card import _get_card_entities
from src.rl.encoding.card import card_key
from src.rl.encoding.effect import effect_key
from src.rl.encoding.focus import derive_focus
from src.rl.encoding.focus import draw_order_visible
from src.rl.encoding.potion import SLICE_KINDS_POTION
from src.rl.encoding.relic import SLICE_KINDS_RELIC
from src.rl.types import ACTION_TYPE_POOL
from src.rl.types import AT_MONSTER_TARGETED
from src.rl.types import AT_REWARD_CARD
from src.rl.types import NUM_ACTION_TYPES
from src.rl.types import SliceKind
from src.rl.types import X_EntityMask
from src.rl.types import X_GameState
from src.rl.types import X_Mask
from src.rl.utils import card_pile_draw_offset
from src.rl.utils import pending_virtual_int


# The block (X_GameState / X_Mask field) each kind's tokens live in
_BLOCK_OF_KIND: dict[SliceKind, str] = {
    **{kind: "cards" for kind in SLICE_KINDS_CARD},
    **{kind: "relics" for kind in SLICE_KINDS_RELIC},
    **{kind: "potions" for kind in SLICE_KINDS_POTION},
    SliceKind.MONSTERS: "monsters",
    SliceKind.EVENT_OPTIONS: "event_options",
    SliceKind.CHARACTER: "character",
    SliceKind.ROOMS: "rooms",
}
_BLOCKS = ("cards", "relics", "potions", "monsters", "event_options", "character", "rooms")


def card_identity_ids(cards: list[Card]) -> list[int]:
    """Contiguous group ids over a pile; same-identity cards share an id (selection dedup).

    Groups by (card_key, effect_key) — the card row plus the effect rows, the
    card's full model-visible identity. Two same-name cards can carry different
    effects (Glass Knife decay, Ritual Dagger bumps) and are NOT interchangeable
    picks. Not keyed on the Card object: the engine's __hash__ covers the arena
    `id`, which would make every card distinct and the dedup a no-op."""
    seen: dict[tuple, int] = {}
    return [
        seen.setdefault(
            (
                card_key(card),
                effect_key(card.effects_play),
                effect_key(card.effects_discard),
                effect_key(card.effects_draw),
            ),
            len(seen),
        )
        for card in cards
    ]


def _pile_ids(
    state: slai.GameState,
    combat: slai.Combat | None,
    target_kinds: tuple[SliceKind, ...] | None,
    num_packed: int,
) -> list[int]:
    """Identity group-ids for a card target's dedup pile, over its packed tokens;
    [] if the target doesn't dedup.

    Multi-kind pools are the reward bundles and never dedup: the same card
    offered in two bundles is two DIFFERENT actions (taking one consumes its
    whole bundle). `combat` is the caller's hoisted `state.combat`."""
    if target_kinds is None or len(target_kinds) > 1:
        return []
    kind = target_kinds[0]
    if kind not in SLICE_KINDS_CARD:
        return []
    # Frozen Eye shows the draw order: same-identity copies at different depths
    # are different picks
    if kind is SliceKind.CARD_PILE_DRAW and draw_order_visible(state):
        return []
    return card_identity_ids(_get_card_entities(kind, state, combat)[:num_packed])


def _kind_and_index(at: int, action: slai.Action) -> tuple[SliceKind, int]:
    """The (kind, engine index) a legal action's L1 pick names."""
    idxs = action.idxs
    if at in AT_REWARD_CARD:
        # (idx_bundle[, idx_card]): the bundle IS a kind; SingingBowl names it via its first card
        return ACTION_TYPE_POOL[at][idxs[0]], idxs[1] if len(idxs) == 2 else 0
    return ACTION_TYPE_POOL[at][0], idxs[0]


# Action types already warned about dropped selections — truncation is never silent
_WARNED_DROPPED: set[int] = set()


def build_masks(
    states: list[slai.GameState],
    legal_actions_batch: list[list],
    x_game_state: X_GameState,
    device: torch.device,
) -> X_Mask:
    """Build a X_Mask from each state's legal actions, one X_EntityMask per block.

    A pick's position in its block is the engine-index-th token of its kind in
    that block's kind row, so the masks read the encoder's placement rather than
    re-deriving it, and an entity the encoder dropped is unselectable by
    construction. The rooms block is laid out by map column, which is what the
    engine names a room by; the monsters block is dense, so a monster's
    position is its engine index."""
    B = len(states)
    rows = {name: getattr(x_game_state, name).x_slice_kind.cpu().numpy() for name in _BLOCKS[:-1]}
    rows["rooms"] = x_game_state.room_idxs.cpu().numpy()
    np_action_type = np.zeros((B, NUM_ACTION_TYPES), dtype=bool)
    np_legal = {
        name: np.zeros((B, NUM_ACTION_TYPES, row.shape[1]), dtype=bool)
        for name, row in rows.items()
    }
    np_needs_target = {name: np.zeros((B, row.shape[1]), dtype=bool) for name, row in rows.items()}
    np_is_target = {name: np.zeros((B, row.shape[1]), dtype=bool) for name, row in rows.items()}

    for b, state in enumerate(states):
        combat = state.combat  # hoisted once: FFI context accesses deep-clone
        pending_at = pending_virtual_int(state)  # likewise: one halt per state
        offset_draw = None  # the draw pile's offset, read on its first pick

        # Block positions of each kind's tokens, in engine order
        positions = {
            kind: np.flatnonzero(rows[block][b] == kind) for kind, block in _BLOCK_OF_KIND.items()
        }
        num_monsters = len(positions[SliceKind.MONSTERS])

        for action in legal_actions_batch[b]:
            # Fold the halt resolver onto its pool's virtual type (types.PENDING_POOL_INT)
            if action.action_type is ActionType.EffectPendingResolve:
                if pending_at is None:
                    raise RuntimeError(
                        "EffectPendingResolve is legal but state.effect_pending is None"
                    )
                at = pending_at
            else:
                at = int(action.action_type)
            idxs = action.idxs
            if at in ACTION_TYPE_POOL and idxs:
                kind, i = _kind_and_index(at, action)
                block = _BLOCK_OF_KIND[kind]
                if kind is SliceKind.ROOMS:  # the rooms block is by column: the index IS the slot
                    p = i if rows[block][b, i] >= 0 else None
                else:
                    rank = i
                    if kind is SliceKind.CARD_PILE_DRAW:
                        # The draw pile's tokens are its top cards: rank = index - offset,
                        # and a card below the offset has no token
                        if offset_draw is None:
                            assert combat is not None, "a draw-pile pick outside combat"
                            offset_draw = card_pile_draw_offset(combat.card_pile_draw)
                        rank -= offset_draw
                    p = positions[kind][rank] if 0 <= rank < len(positions[kind]) else None
                if p is None:
                    # No token (cap overflow, or the encoder disagrees with the engine):
                    # unselectable, so its type bit stays off
                    if at not in _WARNED_DROPPED:
                        _WARNED_DROPPED.add(at)
                        warnings.warn(
                            f"Dropped legal action: type={at} {kind.name}[{i}] has no token"
                        )
                    continue
                np_legal[block][b, at, p] = True  # per ACTION TYPE — legality differs per type
                # L3: only CardPlay/PotionUse carry a monster target as idxs[1]. The
                # engine offers every alive monster to every targeted action, so the
                # targets live on the monster tokens and the pick only records that
                # it takes one
                if at in AT_MONSTER_TARGETED and len(idxs) == 2 and idxs[1] < num_monsters:
                    np_is_target["monsters"][b, idxs[1]] = True
                    np_needs_target[block][b, p] = True
            np_action_type[b, at] = True  # ActionType: every legal kind the masks can express
        if not np_action_type[b].any():
            # Every legal action was dropped — unrepresentable; fail loudly not NaN
            raise RuntimeError(
                f"All legal actions dropped on focus {derive_focus(state).name}: "
                f"{[(int(a.action_type), list(a.idxs)) for a in legal_actions_batch[b]]}"
            )

        # Dedup each type's row by card identity (copies collapse); non-dedup pools return []
        for at, target_kinds in ACTION_TYPE_POOL.items():
            np_row = np_legal[_BLOCK_OF_KIND[target_kinds[0]]][b, at]
            if not np_row.any():
                continue
            kind_positions = positions[target_kinds[0]]
            ids = _pile_ids(state, combat, target_kinds, len(kind_positions))
            if not ids:
                continue
            seen: set = set()
            for i, p in enumerate(kind_positions):
                if not np_row[p]:
                    continue
                if ids[i] in seen:
                    np_row[p] = False
                else:
                    seen.add(ids[i])

    def block_mask(name: str) -> X_EntityMask:
        return X_EntityMask(
            legal=torch.from_numpy(np_legal[name]),
            needs_target=torch.from_numpy(np_needs_target[name]),
            is_target=torch.from_numpy(np_is_target[name]),
            batch_size=[B],
        )

    return X_Mask(
        mask_action_type=torch.from_numpy(np_action_type),
        **{name: block_mask(name) for name in _BLOCKS},
        batch_size=[B],
    ).to(device)
