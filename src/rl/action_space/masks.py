"""
Mask generation for the actor-critic model.

Generates:
- primary_mask: (B, NUM_ACTION_CHOICES) - which ActionChoices are valid
- secondary_masks: {HeadType: (B, output_size)} - per-head entity masks
"""

import numpy as np
import torch

from src.game.const import MAP_WIDTH
from src.game.const import MAX_MONSTERS
from src.game.const import MAX_SIZE_COMBAT_CARD_REWARD
from src.game.const import MAX_SIZE_DECK
from src.game.const import MAX_SIZE_HAND
from src.game.view.fsm import ViewFSM
from src.game.view.state import ViewGameState
from src.rl.action_space.types import ActionChoice
from src.rl.action_space.types import HeadType
from src.rl.action_space.types import NUM_ACTION_CHOICES
from src.rl.action_space.types import NUM_HEAD_TYPES


# =============================================================================
# Mask Sizes (for pre-allocation)
# =============================================================================

_HEAD_TYPE_SIZES = {
    HeadType.CARD_PLAY: MAX_SIZE_HAND,
    HeadType.CARD_DISCARD: MAX_SIZE_HAND,
    HeadType.CARD_REWARD_SELECT: MAX_SIZE_COMBAT_CARD_REWARD,
    HeadType.CARD_UPGRADE: MAX_SIZE_DECK,
    HeadType.MONSTER_SELECT: MAX_MONSTERS,
    HeadType.MAP_SELECT: MAP_WIDTH,
}


# =============================================================================
# Single State Mask Functions (for get_masks)
# =============================================================================


def _get_primary_mask(state: ViewGameState) -> list[bool]:
    """Get valid ActionChoice mask based on game state."""
    mask = [False] * NUM_ACTION_CHOICES

    match state.fsm:
        case ViewFSM.COMBAT_DEFAULT:
            mask[ActionChoice.COMBAT_TURN_END] = True
            has_playable = any(card.cost <= state.energy.current for card in state.hand)
            mask[ActionChoice.CARD_PLAY] = has_playable

        case ViewFSM.COMBAT_AWAIT_TARGET_CARD:
            mask[ActionChoice.MONSTER_SELECT] = True

        case ViewFSM.COMBAT_AWAIT_TARGET_DISCARD:
            mask[ActionChoice.CARD_DISCARD] = True

        case ViewFSM.CARD_REWARD:
            mask[ActionChoice.CARD_REWARD_SKIP] = True
            if len(state.deck) < MAX_SIZE_DECK and state.reward_combat:
                mask[ActionChoice.CARD_REWARD_SELECT] = True

        case ViewFSM.REST_SITE:
            mask[ActionChoice.REST_SITE_REST] = True
            has_upgradable = any(not card.name.endswith("+") for card in state.deck)
            mask[ActionChoice.CARD_UPGRADE] = has_upgradable

        case ViewFSM.MAP:
            mask[ActionChoice.MAP_SELECT] = True

    return mask


def _get_mask_card_play(state: ViewGameState) -> list[bool]:
    """Mask for playable cards in hand."""
    mask = [False] * MAX_SIZE_HAND
    for idx, card in enumerate(state.hand):
        mask[idx] = card.cost <= state.energy.current
    return mask


def _get_mask_card_discard(state: ViewGameState) -> list[bool]:
    """Mask for discardable cards (all cards in hand are valid)."""
    mask = [False] * MAX_SIZE_HAND
    for idx in range(len(state.hand)):
        mask[idx] = True
    return mask


def _get_mask_card_reward(state: ViewGameState) -> list[bool]:
    """Mask for selectable reward cards."""
    mask = [False] * MAX_SIZE_COMBAT_CARD_REWARD
    for idx in range(len(state.reward_combat)):
        mask[idx] = True
    return mask


def _get_mask_card_upgrade(state: ViewGameState) -> list[bool]:
    """Mask for upgradable cards in deck."""
    mask = [False] * MAX_SIZE_DECK
    for idx, card in enumerate(state.deck):
        mask[idx] = not card.name.endswith("+")
    return mask


def _get_mask_monster(state: ViewGameState) -> list[bool]:
    """Mask for targetable monsters."""
    mask = [False] * MAX_MONSTERS
    for idx in range(len(state.monsters)):
        mask[idx] = True
    return mask


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


_SECONDARY_MASK_FNS: dict[HeadType, callable] = {
    HeadType.CARD_PLAY: _get_mask_card_play,
    HeadType.CARD_DISCARD: _get_mask_card_discard,
    HeadType.CARD_REWARD_SELECT: _get_mask_card_reward,
    HeadType.CARD_UPGRADE: _get_mask_card_upgrade,
    HeadType.MONSTER_SELECT: _get_mask_monster,
    HeadType.MAP_SELECT: _get_mask_map,
}


def _get_secondary_mask(head_type: HeadType, state: ViewGameState) -> list[bool]:
    """Get mask for a specific secondary head."""
    return _SECONDARY_MASK_FNS[head_type](state)


# =============================================================================
# Batch Mask Functions (NumPy pre-allocated)
# =============================================================================


def _fill_primary_mask_batch(out: np.ndarray, states: list[ViewGameState]) -> None:
    """Fill pre-allocated primary mask array for batch of states."""
    for b, state in enumerate(states):
        match state.fsm:
            case ViewFSM.COMBAT_DEFAULT:
                out[b, ActionChoice.COMBAT_TURN_END] = True
                has_playable = any(card.cost <= state.energy.current for card in state.hand)
                out[b, ActionChoice.CARD_PLAY] = has_playable

            case ViewFSM.COMBAT_AWAIT_TARGET_CARD:
                out[b, ActionChoice.MONSTER_SELECT] = True

            case ViewFSM.COMBAT_AWAIT_TARGET_DISCARD:
                out[b, ActionChoice.CARD_DISCARD] = True

            case ViewFSM.CARD_REWARD:
                out[b, ActionChoice.CARD_REWARD_SKIP] = True
                if len(state.deck) < MAX_SIZE_DECK and state.reward_combat:
                    out[b, ActionChoice.CARD_REWARD_SELECT] = True

            case ViewFSM.REST_SITE:
                out[b, ActionChoice.REST_SITE_REST] = True
                has_upgradable = any(not card.name.endswith("+") for card in state.deck)
                out[b, ActionChoice.CARD_UPGRADE] = has_upgradable

            case ViewFSM.MAP:
                out[b, ActionChoice.MAP_SELECT] = True


def _fill_secondary_masks_batch(
    out_dict: dict[HeadType, np.ndarray],
    states: list[ViewGameState],
) -> None:
    """Fill pre-allocated secondary mask arrays for batch of states."""
    out_card_play = out_dict[HeadType.CARD_PLAY]
    out_card_discard = out_dict[HeadType.CARD_DISCARD]
    out_card_reward = out_dict[HeadType.CARD_REWARD_SELECT]
    out_card_upgrade = out_dict[HeadType.CARD_UPGRADE]
    out_monster = out_dict[HeadType.MONSTER_SELECT]
    out_map = out_dict[HeadType.MAP_SELECT]

    for b, state in enumerate(states):
        # Card play: playable cards in hand
        for idx, card in enumerate(state.hand):
            out_card_play[b, idx] = card.cost <= state.energy.current

        # Card discard: all cards in hand
        for idx in range(len(state.hand)):
            out_card_discard[b, idx] = True

        # Card reward: available reward cards
        for idx in range(len(state.reward_combat)):
            out_card_reward[b, idx] = True

        # Card upgrade: upgradable cards in deck
        for idx, card in enumerate(state.deck):
            out_card_upgrade[b, idx] = not card.name.endswith("+")

        # Monster select: all monsters
        for idx in range(len(state.monsters)):
            out_monster[b, idx] = True

        # Map select
        if state.map.nodes:
            if state.map.x_current is None and state.map.y_current is None:
                # First floor
                for x, node in enumerate(state.map.nodes[0]):
                    if node is not None:
                        out_map[b, x] = True
            else:
                y = state.map.y_current
                x = state.map.x_current
                if x is not None and x >= 0 and y is not None and y < len(state.map.nodes):
                    row = state.map.nodes[y]
                    if x < len(row):
                        current_node = row[x]
                        if current_node is not None and current_node.x_next:
                            for x_next in current_node.x_next:
                                if 0 <= x_next < MAP_WIDTH:
                                    out_map[b, x_next] = True


# =============================================================================
# Public API
# =============================================================================


def get_masks(
    state: ViewGameState,
    device: torch.device,
) -> tuple[torch.Tensor, dict[HeadType, torch.Tensor]]:
    """
    Get all masks for a single game state.

    Returns:
        primary_mask: (1, NUM_ACTION_CHOICES)
        secondary_masks: {HeadType: (1, head_output_size)}
    """
    primary = _get_primary_mask(state)
    primary_mask = torch.tensor([primary], dtype=torch.bool, device=device)

    secondary_masks = {}
    for head_type in HeadType:
        mask = _get_secondary_mask(head_type, state)
        secondary_masks[head_type] = torch.tensor([mask], dtype=torch.bool, device=device)

    return primary_mask, secondary_masks


def get_masks_batch(
    states: list[ViewGameState],
    device: torch.device,
) -> tuple[torch.Tensor, dict[HeadType, torch.Tensor]]:
    """
    Get all masks for a batch of game states using NumPy pre-allocation.

    Single pass over states fills all masks at once.

    Returns:
        primary_mask: (B, NUM_ACTION_CHOICES)
        secondary_masks: {HeadType: (B, head_output_size)}
    """
    batch_size = len(states)

    # Pre-allocate primary mask
    primary_np = np.zeros((batch_size, NUM_ACTION_CHOICES), dtype=bool)
    _fill_primary_mask_batch(primary_np, states)

    # Pre-allocate all secondary masks
    secondary_np = {
        head_type: np.zeros((batch_size, size), dtype=bool)
        for head_type, size in _HEAD_TYPE_SIZES.items()
    }
    _fill_secondary_masks_batch(secondary_np, states)

    # Convert to tensors
    primary_mask = torch.from_numpy(primary_np).to(device)
    secondary_masks = {
        head_type: torch.from_numpy(arr).to(device)
        for head_type, arr in secondary_np.items()
    }

    return primary_mask, secondary_masks
