import torch

from src.game.const import MAP_WIDTH
from src.game.const import MAX_MONSTERS
from src.game.const import MAX_SIZE_COMBAT_CARD_REWARD
from src.game.const import MAX_SIZE_DECK
from src.game.const import MAX_SIZE_HAND
from src.game.view.state import ViewGameState
from src.rl.action_space.cascade import FSM_ROUTING
from src.rl.action_space.types import HeadType


def _get_mask_action_type(view_game_state: ViewGameState) -> list[bool]:
    """
    Get valid action type mask based on FSM state.
    Returns a mask over the action types in FSM_ROUTING[fsm].action_types.
    """
    route = FSM_ROUTING[view_game_state.fsm]
    mask = [True] * len(route.action_types)

    # Apply additional constraints based on game state
    for idx, action_type in enumerate(route.action_types):
        match action_type.value:
            case "COMBAT_CARD_IN_HAND_SELECT":
                # Can only select a card if there's at least one playable card
                playable = any(
                    card.cost <= view_game_state.energy.current for card in view_game_state.hand
                )
                mask[idx] = playable

            case "CARD_REWARD_SELECT":
                # Can only select if deck isn't full
                mask[idx] = len(view_game_state.deck) < MAX_SIZE_DECK

            case "REST_SITE_UPGRADE":
                # Can only upgrade if there's at least one non-upgraded card
                has_upgradable = any(not card.name.endswith("+") for card in view_game_state.deck)
                mask[idx] = has_upgradable

    return mask


def _get_mask_card_play(view_game_state: ViewGameState) -> list[bool]:
    """Get mask for playable cards in hand."""
    mask = [False] * MAX_SIZE_HAND
    for idx, card in enumerate(view_game_state.hand):
        mask[idx] = card.cost <= view_game_state.energy.current
    return mask


def _get_mask_card_discard(view_game_state: ViewGameState) -> list[bool]:
    """Get mask for discardable cards in hand (all cards are valid)."""
    mask = [False] * MAX_SIZE_HAND
    mask[: len(view_game_state.hand)] = [True] * len(view_game_state.hand)
    return mask


def _get_mask_card_reward_select(view_game_state: ViewGameState) -> list[bool]:
    """Get mask for selectable reward cards."""
    mask = [False] * MAX_SIZE_COMBAT_CARD_REWARD
    mask[: len(view_game_state.reward_combat)] = [True] * len(view_game_state.reward_combat)
    return mask


def _get_mask_card_upgrade(view_game_state: ViewGameState) -> list[bool]:
    """Get mask for upgradable cards in deck."""
    mask = [False] * MAX_SIZE_DECK
    for idx, card in enumerate(view_game_state.deck):
        # Can upgrade if card isn't already upgraded
        mask[idx] = not card.name.endswith("+")
    return mask


def _get_mask_monster_select(view_game_state: ViewGameState) -> list[bool]:
    """Get mask for targetable monsters."""
    mask = [False] * MAX_MONSTERS
    mask[: len(view_game_state.monsters)] = [True] * len(view_game_state.monsters)
    return mask


def _get_mask_map_select(view_game_state: ViewGameState) -> list[bool]:
    """Get mask for selectable map nodes."""
    mask = [False] * MAP_WIDTH

    if view_game_state.map.x_current is None and view_game_state.map.y_current is None:
        # First floor: select from available starting nodes
        for x, node in enumerate(view_game_state.map.nodes[0]):
            if node is not None:
                mask[x] = True
    else:
        # Subsequent floors: select from connected nodes
        current_node = view_game_state.map.nodes[view_game_state.map.y_current][
            view_game_state.map.x_current
        ]
        for x in current_node.x_next:
            mask[x] = True

    return mask


# Registry of mask functions
_MASK_FUNCTIONS = {
    HeadType.ACTION_TYPE: _get_mask_action_type,
    HeadType.CARD_PLAY: _get_mask_card_play,
    HeadType.CARD_DISCARD: _get_mask_card_discard,
    HeadType.CARD_REWARD_SELECT: _get_mask_card_reward_select,
    HeadType.CARD_UPGRADE: _get_mask_card_upgrade,
    HeadType.MONSTER_SELECT: _get_mask_monster_select,
    HeadType.MAP_SELECT: _get_mask_map_select,
}


def get_valid_mask(
    head_type: HeadType,
    view_game_state: ViewGameState,
) -> list[bool]:
    """
    Get the valid action mask for a specific head type.

    Args:
        head_type: Which head we need the mask for
        view_game_state: Current game state

    Returns:
        List of booleans, True = valid action
    """
    return _MASK_FUNCTIONS[head_type](view_game_state)


def get_valid_mask_batch(
    head_type: HeadType,
    view_game_states: list[ViewGameState],
    device: torch.device,
) -> torch.Tensor:
    """
    Get valid action masks for a batch of game states.

    Args:
        head_type: Which head we need masks for
        view_game_states: Batch of game states
        device: Torch device

    Returns:
        Boolean tensor of shape (batch_size, num_options)
    """
    masks = [get_valid_mask(head_type, state) for state in view_game_states]
    return torch.tensor(masks, dtype=torch.bool, device=device)
