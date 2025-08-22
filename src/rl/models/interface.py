import torch

from src.game.action import Action
from src.game.action import ActionType
from src.game.const import CARD_REWARD_NUM
from src.game.const import MAP_WIDTH
from src.game.const import MAX_MONSTERS
from src.game.const import MAX_SIZE_DECK
from src.game.const import MAX_SIZE_HAND
from src.game.view.fsm import ViewFSM
from src.game.view.state import ViewGameState


def get_valid_action_mask(view_game_state: ViewGameState) -> torch.Tensor:
    # Initialize all possible actions as invalid
    values_card_reward_select = torch.zeros(CARD_REWARD_NUM, dtype=torch.bool)
    values_card_reward_skip = torch.zeros(1, dtype=torch.bool)
    values_combat_card_in_hand_select = torch.zeros(MAX_SIZE_HAND, dtype=torch.bool)
    values_combat_card_in_hand_discard = torch.zeros(MAX_SIZE_HAND, dtype=torch.bool)
    values_combat_monster_select = torch.zeros(MAX_MONSTERS, dtype=torch.bool)
    values_combat_turn_end = torch.zeros(1, dtype=torch.bool)
    values_map_node_select = torch.zeros(MAP_WIDTH, dtype=torch.bool)
    values_rest_site_rest = torch.zeros(1, dtype=torch.bool)
    values_rest_site_upgrade = torch.zeros(MAX_SIZE_DECK, dtype=torch.bool)

    match view_game_state.fsm:
        case ViewFSM.CARD_REWARD:
            if len(view_game_state.deck) < MAX_SIZE_DECK:
                values_card_reward_select[:] = True

            values_card_reward_skip[0] = True

        case ViewFSM.COMBAT_DEFAULT:
            # Mark playable card indexes as valid for selection TODO: uplayable cards, e.g., "Injury"
            for idx, view_card in enumerate(view_game_state.hand):
                if view_card.cost <= view_game_state.energy.current:
                    values_combat_card_in_hand_select[idx] = True

            # Mark turn end as valid
            values_combat_turn_end[0] = True

        case ViewFSM.COMBAT_AWAIT_TARGET_CARD:
            # Mark monsters as valid for selection
            values_combat_monster_select[: len(view_game_state.monsters)] = True

        case ViewFSM.COMBAT_AWAIT_TARGET_DISCARD:
            values_combat_card_in_hand_discard[: len(view_game_state.hand)] = True

        case ViewFSM.MAP:
            if view_game_state.map.x_current is None and view_game_state.map.y_current is None:
                # Need to select from the first floor
                for x, node in enumerate(view_game_state.map.nodes[0]):
                    if node is not None:
                        values_map_node_select[x] = True

            else:
                # Need to select from next floor
                map_node_active = view_game_state.map.nodes[view_game_state.map.y_current][
                    view_game_state.map.x_current
                ]
                # TODO: this will probably raise an error when the game is over
                values_map_node_select[list(map_node_active.x_next)] = True

        case ViewFSM.REST_SITE:
            values_rest_site_rest[0] = True

            # Mark unupgraded cards as valid for selection
            for idx, view_card in enumerate(view_game_state.deck):
                # TODO: add `upgraded` field
                if view_card.name.endswith("+"):
                    continue

                values_rest_site_upgrade[idx] = True

    return torch.cat(
        (
            values_card_reward_select,
            values_card_reward_skip,
            values_combat_card_in_hand_select,
            values_combat_card_in_hand_discard,
            values_combat_monster_select,
            values_combat_turn_end,
            values_map_node_select,
            values_rest_site_rest,
            values_rest_site_upgrade,
        ),
        axis=0,
    )


# TODO: try to make code prettier
def action_idx_to_action(action_idx: int) -> Action:
    if action_idx < 0:
        raise ValueError("Argument `action_idx` can't be negative")

    if action_idx < CARD_REWARD_NUM:
        return Action(ActionType.CARD_REWARD_SELECT, action_idx)

    if action_idx < CARD_REWARD_NUM + 1:
        return Action(ActionType.CARD_REWARD_SKIP)

    if action_idx < CARD_REWARD_NUM + 1 + MAX_SIZE_HAND:
        return Action(ActionType.COMBAT_CARD_IN_HAND_SELECT, action_idx - CARD_REWARD_NUM - 1)

    if action_idx < CARD_REWARD_NUM + 1 + MAX_SIZE_HAND + MAX_SIZE_HAND:
        return Action(
            ActionType.COMBAT_CARD_IN_HAND_SELECT, action_idx - CARD_REWARD_NUM - 1 - MAX_SIZE_HAND
        )

    if action_idx < CARD_REWARD_NUM + 1 + MAX_SIZE_HAND + MAX_SIZE_HAND + MAX_MONSTERS:
        return Action(
            ActionType.COMBAT_MONSTER_SELECT,
            action_idx - CARD_REWARD_NUM - 1 - MAX_SIZE_HAND - MAX_SIZE_HAND,
        )

    if action_idx < CARD_REWARD_NUM + 1 + MAX_SIZE_HAND + MAX_SIZE_HAND + MAX_MONSTERS + 1:
        return Action(ActionType.COMBAT_TURN_END)

    if (
        action_idx
        < CARD_REWARD_NUM + 1 + MAX_SIZE_HAND + MAX_SIZE_HAND + MAX_MONSTERS + 1 + MAP_WIDTH
    ):
        return Action(
            ActionType.MAP_NODE_SELECT,
            action_idx - CARD_REWARD_NUM - 1 - MAX_SIZE_HAND - MAX_SIZE_HAND - MAX_MONSTERS - 1,
        )

    if (
        action_idx
        < CARD_REWARD_NUM + 1 + MAX_SIZE_HAND + MAX_SIZE_HAND + MAX_MONSTERS + 1 + MAP_WIDTH + 1
    ):
        return Action(ActionType.REST_SITE_REST)

    if (
        action_idx
        < CARD_REWARD_NUM
        + 1
        + MAX_SIZE_HAND
        + MAX_SIZE_HAND
        + MAX_MONSTERS
        + 1
        + MAP_WIDTH
        + 1
        + MAX_SIZE_DECK
    ):
        return Action(
            ActionType.REST_SITE_UPGRADE,
            action_idx
            - CARD_REWARD_NUM
            - 1
            - MAX_SIZE_HAND
            - MAX_SIZE_HAND
            - MAX_MONSTERS
            - 1
            - MAP_WIDTH
            - 1,
        )

    raise ValueError(f"Unsupported `action_idx`: {action_idx}")
