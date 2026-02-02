"""
Mask generation for the actor-critic model.

Generates:
- primary_mask: (B, NUM_ACTION_CHOICES) - which ActionChoices are valid
- secondary_masks: {HeadType: (B, output_size)} - per-head entity masks
"""

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


# =============================================================================
# Primary Mask (ActionChoice)
# =============================================================================


def _get_primary_mask(state: ViewGameState) -> list[bool]:
    """
    Get valid ActionChoice mask based on game state.

    Returns a mask of size NUM_ACTION_CHOICES.
    """
    mask = [False] * NUM_ACTION_CHOICES

    match state.fsm:
        case ViewFSM.COMBAT_DEFAULT:
            # Can end turn
            mask[ActionChoice.COMBAT_TURN_END] = True

            # Can play a card if any card is affordable
            has_playable = any(card.cost <= state.energy.current for card in state.hand)
            mask[ActionChoice.CARD_PLAY] = has_playable

        case ViewFSM.COMBAT_AWAIT_TARGET_CARD:
            # Must select a monster target
            mask[ActionChoice.MONSTER_SELECT] = True

        case ViewFSM.COMBAT_AWAIT_TARGET_DISCARD:
            # Must discard a card
            mask[ActionChoice.CARD_DISCARD] = True

        case ViewFSM.CARD_REWARD:
            # Can skip
            mask[ActionChoice.CARD_REWARD_SKIP] = True

            # Can select a card if deck isn't full
            if len(state.deck) < MAX_SIZE_DECK and state.reward_combat:
                mask[ActionChoice.CARD_REWARD_SELECT] = True

        case ViewFSM.REST_SITE:
            # Can rest
            mask[ActionChoice.REST_SITE_REST] = True

            # Can upgrade if there's an upgradable card
            has_upgradable = any(not card.name.endswith("+") for card in state.deck)
            mask[ActionChoice.CARD_UPGRADE] = has_upgradable

        case ViewFSM.MAP:
            # Must select a map node
            mask[ActionChoice.MAP_SELECT] = True

    return mask


# =============================================================================
# Secondary Masks (per HeadType)
# =============================================================================


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

    # Handle case where map state is not valid for selection
    if not state.map.nodes:
        return mask

    if state.map.x_current is None and state.map.y_current is None:
        # First floor: available starting nodes
        for x, node in enumerate(state.map.nodes[0]):
            if node is not None:
                mask[x] = True
    else:
        y = state.map.y_current
        x = state.map.x_current

        # Boss floor has x=-1, no further map selection possible
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
    Get all masks for a batch of game states.

    Returns:
        primary_mask: (B, NUM_ACTION_CHOICES)
        secondary_masks: {HeadType: (B, head_output_size)}
    """
    primary_masks = [_get_primary_mask(s) for s in states]
    primary_mask = torch.tensor(primary_masks, dtype=torch.bool, device=device)

    secondary_masks = {}
    for head_type in HeadType:
        masks = [_get_secondary_mask(head_type, s) for s in states]
        secondary_masks[head_type] = torch.tensor(masks, dtype=torch.bool, device=device)

    return primary_mask, secondary_masks
