import warnings

import numpy as np
import slai
import torch
from tensordict import TensorDict

from src.rl.index import TOKEN_SIZE
from src.rl.index import Token
from src.rl.index import TokenKind
from src.rl.index import token_entities
from src.rl.types import AT_SIZE
from src.rl.types import AT_TARGET
from src.rl.types import NUM_ACTION_TYPES
from src.rl.types import SELECTING_ACTION_TYPES
from src.rl.types import TMask
from src.rl.constants import MAX_MONSTERS
from src.rl.constants import MAX_POTION_SLOTS
from src.rl.constants import MAX_SIZE_HAND


def card_identity_ids(cards: list[slai.Card]) -> list[int]:
    """Contiguous group ids (from 0) over a pile; cards with the same identity
    key share an id (selection dedup). One id per card; the caller pads
    truncated/empty slots with -1."""
    seen: dict[tuple, int] = {}
    return [seen.setdefault(card.identity_hash, len(seen)) for card in cards]


def _pile_ids(state: slai.GameState, target: "Token | object") -> list[int]:
    """Identity group-ids for a card target's dedup source pile; [] if the target doesn't
    dedup (only card piles dedup — copies of the same card collapse to one selectable slot).
    The dedup pile is exactly the target token's entities (truncated to its cap)."""
    if not isinstance(target, Token) or target.kind != TokenKind.CARD:
        return []
    return card_identity_ids(token_entities(target, state)[: TOKEN_SIZE[target]])


# Action types we've already warned about dropping (selection index beyond the
# encoder's pool cap) — truncation must never be silent.
_WARNED_DROPPED: set[int] = set()


def build_masks(
    states: list[slai.GameState],
    legal_actions_batch: list[list],
    device: torch.device,
) -> TMask:
    """Build a TMask from each state's engine-emitted legal actions. Every legal
    action type sets its L1 bit (a halt ends up with exactly one); each action type's
    legal entities go in its own L2 mask (legality is per type, deduped per pool)."""
    B = len(states)
    action_type_np = np.zeros((B, NUM_ACTION_TYPES), dtype=bool)
    sel_np = {at: np.zeros((B, AT_SIZE[at]), dtype=bool) for at in SELECTING_ACTION_TYPES}
    card_tgt_np = np.zeros((B, MAX_SIZE_HAND, MAX_MONSTERS), dtype=bool)
    potion_tgt_np = np.zeros((B, MAX_POTION_SLOTS, MAX_MONSTERS), dtype=bool)

    for i, state in enumerate(states):
        for action in legal_actions_batch[i]:
            at = int(action.action_type)
            idxs = action.idxs
            target = AT_TARGET[at]
            if target is not None and idxs:
                if idxs[0] >= AT_SIZE[at]:
                    # Beyond the encoder cap: unselectable, so don't legalize its
                    # L1 kind off it either (a legal kind with an all-False L2 mask
                    # would NaN the selection Categorical).
                    if at not in _WARNED_DROPPED:
                        _WARNED_DROPPED.add(at)
                        warnings.warn(
                            f"Dropped legal action: type={at} idx={idxs[0]} exceeds "
                            f"pool cap {AT_SIZE[at]} (see constants.MAX_SIZE_*)"
                        )
                    continue
                sel_np[at][i, idxs[0]] = True  # per ACTION TYPE — legality differs per type
                if len(idxs) == 2 and idxs[1] < MAX_MONSTERS:
                    if target.kind == TokenKind.CARD:
                        card_tgt_np[i, idxs[0], idxs[1]] = True
                    elif target.kind == TokenKind.POTION:
                        potion_tgt_np[i, idxs[0], idxs[1]] = True
            action_type_np[i, at] = True  # L1: every legal kind the masks can express
        if not action_type_np[i].any():
            # Every legal action was beyond a pool cap — unrepresentable state;
            # fail loudly instead of sampling NaNs.
            raise RuntimeError(
                f"All legal actions exceed encoder pool caps on screen {state.screen}: "
                f"{[(int(a.action_type), list(a.idxs)) for a in legal_actions_batch[i]]}"
            )

        # Dedup each action type's selection mask via its pool's identity pile: keep
        # only the first occurrence of each card identity (copies collapse to one).
        # Non-deduping pools return [] from _pile_ids, so the empty check skips them.
        for at in SELECTING_ACTION_TYPES:
            target = AT_TARGET[at]
            row = sel_np[at][i]
            if not row.any():
                continue
            ids = _pile_ids(state, target)
            if not ids:
                continue
            seen: set = set()
            for slot in range(min(len(ids), AT_SIZE[at])):
                if not row[slot]:
                    continue
                if ids[slot] in seen:
                    row[slot] = False
                else:
                    seen.add(ids[slot])

    return TMask(
        mask_action_type=torch.from_numpy(action_type_np),
        mask_action_idx=TensorDict(
            {str(at): torch.from_numpy(sel_np[at]) for at in SELECTING_ACTION_TYPES},
            batch_size=[B],
        ),
        mask_target_card=torch.from_numpy(card_tgt_np),
        mask_target_potion=torch.from_numpy(potion_tgt_np),
        batch_size=[B],
    ).to(device)
