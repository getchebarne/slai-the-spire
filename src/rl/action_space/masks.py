"""
Mask generation for the actor-critic model.

Generates per-HeadTypePrimary masks:
- primary_masks: binary decision masks for decision primaries
- selection_masks: entity selection masks for all primaries
"""

from dataclasses import dataclass

import numpy as np
import torch

from src.game.const import MAP_WIDTH
from src.game.const import MAX_MONSTERS
from src.game.const import MAX_SIZE_COMBAT_CARD_REWARD
from src.game.const import MAX_SIZE_DECK
from src.game.const import MAX_SIZE_HAND
from src.game.view.state import ViewGameState
from src.rl.action_space.route import get_route_primary
from src.rl.action_space.types import DECISION_PRIMARIES
from src.rl.action_space.types import HeadTypePrimary
from src.rl.action_space.types import PRIMARY_NUM_CHOICES


# =============================================================================
# Constants
# =============================================================================

# Entity selection mask sizes per primary type
SELECTION_SIZES: dict[HeadTypePrimary, int] = {
    HeadTypePrimary.COMBAT_DEFAULT: MAX_SIZE_HAND,
    HeadTypePrimary.CARD_REWARD: MAX_SIZE_COMBAT_CARD_REWARD,
    HeadTypePrimary.REST_SITE: MAX_SIZE_DECK,
    HeadTypePrimary.COMBAT_CARD_DISCARD: MAX_SIZE_HAND,
    HeadTypePrimary.COMBAT_MONSTER_SELECT: MAX_MONSTERS,
    HeadTypePrimary.MAP_SELECT: MAP_WIDTH,
}


# =============================================================================
# MaskBatch
# =============================================================================


@dataclass
class MaskBatch:
    """
    Batched masks organized by HeadTypePrimary.

    Attributes:
        route: sample indices per primary group. {type: (N_group,) int64}
        primary_masks: decision masks (decision primaries only). {type: (N_group, num_choices)}
        selection_masks: entity selection masks (all groups). {type: (N_group, max_entities)}
    """

    route: dict[HeadTypePrimary, torch.Tensor]
    primary_masks: dict[HeadTypePrimary, torch.Tensor]
    selection_masks: dict[HeadTypePrimary, torch.Tensor]


# =============================================================================
# Per-state primary mask functions (decision primaries only)
# =============================================================================


def _get_primary_mask_combat_default(state: ViewGameState) -> list[bool]:
    """[end_turn, play_card]"""
    has_playable = any(card.cost <= state.energy.current for card in state.hand)
    return [True, has_playable]


def _get_primary_mask_card_reward(state: ViewGameState) -> list[bool]:
    """[skip, select]"""
    can_select = len(state.deck) < MAX_SIZE_DECK and len(state.reward_combat) > 0
    return [True, can_select]


def _get_primary_mask_rest_site(state: ViewGameState) -> list[bool]:
    """[rest, upgrade]"""
    has_upgradable = any(not card.name.endswith("+") for card in state.deck)
    return [True, has_upgradable]


_PRIMARY_MASK_FNS: dict[HeadTypePrimary, callable] = {
    HeadTypePrimary.COMBAT_DEFAULT: _get_primary_mask_combat_default,
    HeadTypePrimary.CARD_REWARD: _get_primary_mask_card_reward,
    HeadTypePrimary.REST_SITE: _get_primary_mask_rest_site,
}


# =============================================================================
# Per-state selection mask functions (all primaries)
# =============================================================================


def _get_selection_mask(head_type: HeadTypePrimary, state: ViewGameState) -> list[bool]:
    """Get entity selection mask for a primary type."""
    match head_type:
        case HeadTypePrimary.COMBAT_DEFAULT:
            mask = [False] * MAX_SIZE_HAND
            for idx, card in enumerate(state.hand):
                mask[idx] = card.cost <= state.energy.current
            return mask

        case HeadTypePrimary.COMBAT_CARD_DISCARD:
            mask = [False] * MAX_SIZE_HAND
            for idx in range(len(state.hand)):
                mask[idx] = True
            return mask

        case HeadTypePrimary.CARD_REWARD:
            mask = [False] * MAX_SIZE_COMBAT_CARD_REWARD
            for idx in range(len(state.reward_combat)):
                mask[idx] = True
            return mask

        case HeadTypePrimary.REST_SITE:
            mask = [False] * MAX_SIZE_DECK
            for idx, card in enumerate(state.deck):
                mask[idx] = not card.name.endswith("+")
            return mask

        case HeadTypePrimary.COMBAT_MONSTER_SELECT:
            mask = [False] * MAX_MONSTERS
            for idx in range(len(state.monsters)):
                mask[idx] = True
            return mask

        case HeadTypePrimary.MAP_SELECT:
            return _get_mask_map(state)

        case _:
            raise ValueError(f"Unknown head type: {head_type}")


def _get_mask_map(state: ViewGameState) -> list[bool]:
    """Mask for selectable map nodes."""
    mask = [False] * MAP_WIDTH

    if not state.map.nodes:
        return mask

    if state.map.x_current is None and state.map.y_current is None:
        for x, node in enumerate(state.map.nodes[0]):
            if node is not None:
                mask[x] = True
    else:
        y = state.map.y_current
        x = state.map.x_current

        if x is None or x < 0 or y is None or y >= len(state.map.nodes):
            return mask

        row = state.map.nodes[y]
        if x >= len(row):
            return mask

        current_node = row[x]
        if current_node is not None and current_node.x_next:
            for x_next in current_node.x_next:
                if 0 <= x_next < MAP_WIDTH:
                    mask[x_next] = True

    return mask


# =============================================================================
# Public API
# =============================================================================


def get_mask_batch(
    states: list[ViewGameState],
    device: torch.device,
) -> MaskBatch:
    """
    Build MaskBatch for a list of game states.

    Routes states by FSM, then generates primary and selection masks
    for each HeadTypePrimary group using NumPy pre-allocation.
    """
    route_lists = get_route_primary(states)

    route: dict[HeadTypePrimary, torch.Tensor] = {}
    primary_masks: dict[HeadTypePrimary, torch.Tensor] = {}
    selection_masks: dict[HeadTypePrimary, torch.Tensor] = {}

    for head_type in HeadTypePrimary:
        indices = route_lists[head_type]
        route[head_type] = torch.tensor(indices, dtype=torch.long, device=device)

        n = len(indices)
        sel_size = SELECTION_SIZES[head_type]

        if n == 0:
            # Empty group — still create correctly shaped empty tensors
            if head_type in DECISION_PRIMARIES:
                primary_masks[head_type] = torch.zeros(
                    0, PRIMARY_NUM_CHOICES[head_type], dtype=torch.bool, device=device
                )
            selection_masks[head_type] = torch.zeros(0, sel_size, dtype=torch.bool, device=device)
            continue

        group_states = [states[i] for i in indices]

        # Primary masks (decision primaries only)
        if head_type in DECISION_PRIMARIES:
            num_choices = PRIMARY_NUM_CHOICES[head_type]
            primary_fn = _PRIMARY_MASK_FNS[head_type]
            primary_np = np.zeros((n, num_choices), dtype=bool)
            for b, state in enumerate(group_states):
                primary_np[b] = primary_fn(state)
            primary_masks[head_type] = torch.from_numpy(primary_np).to(device)

        # Selection masks (all groups)
        sel_np = np.zeros((n, sel_size), dtype=bool)
        for b, state in enumerate(group_states):
            sel_np[b] = _get_selection_mask(head_type, state)
        selection_masks[head_type] = torch.from_numpy(sel_np).to(device)

    return MaskBatch(
        route=route,
        primary_masks=primary_masks,
        selection_masks=selection_masks,
    )
