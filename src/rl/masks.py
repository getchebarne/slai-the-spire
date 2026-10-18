import warnings
from collections.abc import Sequence

import numpy as np
import slai
import torch
from slai import ACTION_SPEC_REGISTRY
from slai import ActionType
from slai import Card
from slai import EventOption
from slai import Monster
from slai import Potion
from slai import PotionName
from slai import Relic
from slai import RelicName
from slai import Room

from src.rl.constants import MAP_WIDTH
from src.rl.constants import MAX_EVENT_OPTIONS
from src.rl.constants import MAX_MONSTERS
from src.rl.encoding.card import _get_card_entities
from src.rl.encoding.card import card_key
from src.rl.encoding.effect import effect_key
from src.rl.encoding.event import _get_event_option_entities
from src.rl.encoding.focus import derive_focus
from src.rl.encoding.focus import draw_order_visible
from src.rl.encoding.map_ import _get_room_entities
from src.rl.encoding.monster import _get_monster_entities
from src.rl.encoding.potion import _get_potion_entities
from src.rl.encoding.relic import _get_relic_entities
from src.rl.encoding.state import CAP_CARDS
from src.rl.encoding.state import CAP_POTIONS
from src.rl.encoding.state import CAP_RELICS
from src.rl.types import ACTION_TYPE_BY_INT
from src.rl.types import NUM_ACTION_TYPES
from src.rl.types import SLICE_KINDS_CARD
from src.rl.types import SLICE_KINDS_POTION
from src.rl.types import SLICE_KINDS_RELIC
from src.rl.types import SliceKind
from src.rl.types import X_Action
from src.rl.types import X_EntityMask
from src.rl.types import X_GameState
from src.rl.types import X_Mask


# An entity a legal action can name
type Entity = Card | Relic | Potion | Monster | EventOption | Room

# A state's id -> token index: each entity id's token, as its slice kind and rank
type TokenIndex = dict[int, tuple[SliceKind, int]]


def token_entities(
    slice_kind: SliceKind, state: slai.GameState, combat: slai.Combat | None
) -> Sequence[Entity]:
    """The entities of a slice kind's tokens in rank order, from the encoders' own lists, so an
    entity's id and its token share one placement. `combat` is the caller's hoisted
    `state.combat`."""
    if slice_kind in SLICE_KINDS_CARD:
        return _get_card_entities(slice_kind, state, combat)
    if slice_kind in SLICE_KINDS_RELIC:
        return _get_relic_entities(slice_kind, state)
    if slice_kind in SLICE_KINDS_POTION:
        return _get_potion_entities(slice_kind, state)
    match slice_kind:
        case SliceKind.MONSTERS:
            return _get_monster_entities(combat)
        case SliceKind.EVENT_OPTIONS:
            return _get_event_option_entities(state.event)
        case SliceKind.ROOMS:
            return _get_room_entities(state)
        case _:
            raise ValueError(f"No action names a {slice_kind.name} token")


def token_index(state: slai.GameState, combat: slai.Combat | None) -> TokenIndex:
    """Each entity id's token over the window, as its slice kind and rank. An id on several tokens
    names its first in window order, where each block packs its decision pools first (the
    relic event roll before the owned relics it can repeat). `combat` is the caller's
    hoisted `state.combat`."""
    index: TokenIndex = {}
    for slice_kind in (
        *SLICE_KINDS_CARD,
        *SLICE_KINDS_RELIC,
        *SLICE_KINDS_POTION,
        SliceKind.MONSTERS,
        SliceKind.EVENT_OPTIONS,
        SliceKind.ROOMS,
    ):
        # The selected pile and the play queue repeat the ids of cards still in their piles
        if slice_kind in (SliceKind.CARD_PILE_SELECTED, SliceKind.CARD_PLAY_QUEUE):
            continue
        for rank, entity in enumerate(token_entities(slice_kind, state, combat)):
            index.setdefault(entity.id, (slice_kind, rank))
    return index


def card_identity_ids(cards: list[Card]) -> list[int]:
    """Contiguous group ids over a pile; same-identity cards share an id (selection dedup).

    Groups by (card_key, effect_key) — the card row plus the effect rows, the
    card's full model-visible identity. Two same-name cards can carry different
    effects (Glass Knife decay, Ritual Dagger bumps) and are NOT interchangeable
    selections. Not keyed on the Card object: the engine's __hash__ covers the arena
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
    state: slai.GameState, combat: slai.Combat | None, slice_kind: SliceKind
) -> list[int]:
    """Identity group-ids for a card slice kind's dedup, over its packed tokens; [] if the
    slice kind doesn't dedup. `combat` is the caller's hoisted `state.combat`."""
    if slice_kind not in SLICE_KINDS_CARD:
        return []

    # Frozen Eye shows the draw order: same-identity copies at different depths
    # are different selections
    if slice_kind is SliceKind.CARD_PILE_DRAW and draw_order_visible(state):
        return []
    return card_identity_ids(_get_card_entities(slice_kind, state, combat))


def wasted_potion_discard(state: slai.GameState, action: slai.Action) -> bool:
    """True when `action` discards a Potion while nothing needs the room: no Potion can
    arrive under Sozu, and with a free belt slot the discard only pays as a reroll
    before drinking an Entropic Brew held in another slot."""
    if action.action_type is not ActionType.PotionDiscard:
        return False
    if any(relic.name == RelicName.Sozu for relic in state.relics):
        return True
    potions = state.potions
    return len(potions) < state.potion_slots_max and not any(
        potion.name == PotionName.EntropicBrew for potion in potions if potion.id != action.ids[0]
    )


# Action types already warned about dropped selections — truncation is never silent
_WARNED_DROPPED: set[int] = set()


def build_masks(
    states: list[slai.GameState],
    batch_legal_actions: list[list[slai.Action]],
    x_game_state: X_GameState,
    device: torch.device,
) -> X_Mask:
    """Build a X_Mask from each state's legal actions, one X_EntityMask per block.
    Wasted Potion discards are engine-legal but masked.

    A legal id's position in its block is its token's, the rank-th of its slice kind in that
    block's slice kind row, as the state's token_index ranks it; so the masks read the
    encoder's placement rather than re-deriving it, and an entity the encoder dropped
    is unselectable by construction. A target is its monster's token."""
    batch_size = len(states)

    # Read the window's slice kind row: X_GameState's entity blocks, then the rooms row
    x_rows = (
        x_game_state.cards.x_slice_kind,
        x_game_state.relics.x_slice_kind,
        x_game_state.potions.x_slice_kind,
        x_game_state.monsters.x_slice_kind,
        x_game_state.event_options.x_slice_kind,
        x_game_state.character.x_slice_kind,
        x_game_state.room_slice_kind,
    )
    widths = [x_row.shape[1] for x_row in x_rows]
    np_slice_kind = torch.cat(x_rows, dim=1).cpu().numpy()

    # Pre-allocate NumPy arrays
    np_action_type = np.zeros((batch_size, NUM_ACTION_TYPES), dtype=bool)
    np_legal = np.zeros((batch_size, NUM_ACTION_TYPES, np_slice_kind.shape[1]), dtype=bool)
    np_needs_target = np.zeros(np_slice_kind.shape, dtype=bool)
    np_monster_is_target = np.zeros(
        (batch_size, x_game_state.monsters.x_slice_kind.shape[1]), dtype=bool
    )

    for b, state in enumerate(states):
        combat = state.combat  # hoisted once: FFI context accesses deep-clone
        index = token_index(state, combat)

        # Window positions of each slice kind's tokens, in rank order
        positions: dict[SliceKind, np.ndarray] = {}

        # The (type, slice kind) pairs whose legal ids set a bit, for the dedup below
        legal_slice_kinds: set[tuple[int, SliceKind]] = set()

        for action in batch_legal_actions[b]:
            if wasted_potion_discard(state, action):
                continue

            at = int(action.action_type)
            ids = action.ids  # bound once: each read clones the list
            if ids:
                token = index.get(ids[0])
                if token is None:
                    # No token (cap overflow, or the encoder disagrees with the engine):
                    # unselectable, so its type bit stays off
                    if at not in _WARNED_DROPPED:
                        _WARNED_DROPPED.add(at)
                        warnings.warn(f"Dropped legal action: type={at} id {ids[0]} has no token")
                    continue

                # Filled on a slice kind's first use: every one up front costs most of the build
                slice_kind, rank = token
                if slice_kind not in positions:
                    positions[slice_kind] = np.flatnonzero(np_slice_kind[b] == slice_kind)
                w = positions[slice_kind][rank]
                np_legal[b, at, w] = True  # per ACTION TYPE — legality differs per type
                legal_slice_kinds.add((at, slice_kind))

                # L3: only CardPlay/PotionUse carry a monster target as ids[1]. The
                # engine offers every alive monster to every targeted action, so the
                # targets live on the monster tokens and the selection only records that
                # it takes one
                if len(ids) == 2:
                    # The monster block is dense: a monster's rank is its column
                    _, rank_target = index[ids[1]]
                    np_monster_is_target[b, rank_target] = True
                    np_needs_target[b, w] = True
            np_action_type[b, at] = True  # L1: every legal ActionType the masks can express
        if not np_action_type[b].any():
            # Every legal action was dropped — unrepresentable; fail loudly not NaN
            raise RuntimeError(
                f"All legal actions dropped on focus {derive_focus(state).name}: "
                f"{[(int(a.action_type), list(a.ids)) for a in batch_legal_actions[b]]}"
            )

        # Dedup each type's legal tokens by card identity per slice kind (copies collapse)
        for at, slice_kind in legal_slice_kinds:
            group_ids = _pile_ids(state, combat, slice_kind)
            if not group_ids:
                continue

            np_row = np_legal[b, at]
            seen: set[int] = set()
            for rank, w in enumerate(positions[slice_kind]):
                if not np_row[w]:
                    continue
                if group_ids[rank] in seen:
                    np_row[w] = False
                else:
                    seen.add(group_ids[rank])

    # Cut the window back into its blocks: X_GameState's entity blocks, then the rooms
    (
        x_cards_mask,
        x_relics_mask,
        x_potions_mask,
        x_monsters_mask,
        x_event_options_mask,
        x_character_mask,
        x_rooms_mask,
    ) = (
        X_EntityMask(legal=x_legal, needs_target=x_needs_target, batch_size=[batch_size])
        for x_legal, x_needs_target in zip(
            torch.from_numpy(np_legal).split(widths, dim=2),
            torch.from_numpy(np_needs_target).split(widths, dim=1),
        )
    )
    return X_Mask(
        mask_action_type=torch.from_numpy(np_action_type),
        cards=x_cards_mask,
        relics=x_relics_mask,
        potions=x_potions_mask,
        monsters=x_monsters_mask,
        event_options=x_event_options_mask,
        character=x_character_mask,
        rooms=x_rooms_mask,
        monster_is_target=torch.from_numpy(np_monster_is_target),
        batch_size=[batch_size],
    ).to(device)


def empty_masks(batch_size: int) -> X_Mask:
    """Masks with every block at its cap and nothing legal"""
    return X_Mask(
        mask_action_type=torch.zeros(batch_size, NUM_ACTION_TYPES, dtype=torch.bool),
        cards=_empty_entity_mask(batch_size, CAP_CARDS),
        relics=_empty_entity_mask(batch_size, CAP_RELICS),
        potions=_empty_entity_mask(batch_size, CAP_POTIONS),
        monsters=_empty_entity_mask(batch_size, MAX_MONSTERS),
        event_options=_empty_entity_mask(batch_size, MAX_EVENT_OPTIONS),
        character=_empty_entity_mask(batch_size, 1),
        rooms=_empty_entity_mask(batch_size, MAP_WIDTH),
        monster_is_target=torch.zeros(batch_size, MAX_MONSTERS, dtype=torch.bool),
        batch_size=[batch_size],
    )


def _empty_entity_mask(batch_size: int, cap: int) -> X_EntityMask:
    return X_EntityMask(
        legal=torch.zeros(batch_size, NUM_ACTION_TYPES, cap, dtype=torch.bool),
        needs_target=torch.zeros(batch_size, cap, dtype=torch.bool),
        batch_size=[batch_size],
    )


def engine_action(
    at_int: int, slice_kind: int, rank: int, target: int, state: slai.GameState
) -> slai.Action:
    """Build a `slai.Action` from the sampled hierarchy [ActionType, slice kind, rank, target]
    (slice kind/rank = -1 when the type takes no id, target = -1 when it takes no monster
    target).

    The slice kind and rank name a token and the target the target-th monster token; the
    action carries their entities' ids (token_entities)."""
    action_type = ACTION_TYPE_BY_INT[at_int]
    if ACTION_SPEC_REGISTRY[action_type].arity == (0, 0):
        return slai.Action(action_type, [])

    combat = state.combat
    ids = [token_entities(SliceKind(slice_kind), state, combat)[rank].id]
    if target >= 0:  # CardPlay / PotionUse with a monster target
        ids.append(token_entities(SliceKind.MONSTERS, state, combat)[target].id)
    return slai.Action(action_type, ids)


def action_from_actiontype(x_action: X_Action, i: int, state: slai.GameState) -> slai.Action:
    """Decode row `i` of a batched `X_Action` into a `slai.Action`"""
    return engine_action(
        int(x_action.action_type[i]),
        int(x_action.slice_kind[i]),
        int(x_action.rank[i]),
        int(x_action.target[i]),
        state,
    )
