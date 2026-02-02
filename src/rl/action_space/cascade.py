from dataclasses import dataclass

from src.game.action import ActionType
from src.game.core.fsm import FSM
from src.rl.action_space.types import HeadType


@dataclass(frozen=True)
class ActionRoute:
    """Describes the routing for a given FSM state."""

    action_types: tuple[ActionType, ...]

    @property
    def is_forced(self) -> bool:
        """True if there's only one valid action type (skip action type head)."""
        return len(self.action_types) == 1

    @property
    def forced_action_type(self) -> ActionType:
        """The single forced action type. Only valid if is_forced=True."""
        assert self.is_forced, "Not a forced state"
        return self.action_types[0]


# FSM → valid ActionTypes
FSM_ROUTING: dict[FSM, ActionRoute] = {
    # Combat: choose between playing a card or ending turn
    FSM.COMBAT_DEFAULT: ActionRoute(
        action_types=(ActionType.COMBAT_CARD_IN_HAND_SELECT, ActionType.COMBAT_TURN_END),
    ),
    # Combat: forced to select a monster target
    FSM.COMBAT_AWAIT_TARGET_CARD: ActionRoute(
        action_types=(ActionType.COMBAT_MONSTER_SELECT,),
    ),
    # Combat: forced to discard a card
    FSM.COMBAT_AWAIT_TARGET_DISCARD: ActionRoute(
        action_types=(ActionType.COMBAT_CARD_IN_HAND_SELECT,),
    ),
    # Card reward: choose between selecting a card or skipping
    FSM.CARD_REWARD: ActionRoute(
        action_types=(ActionType.CARD_REWARD_SELECT, ActionType.CARD_REWARD_SKIP),
    ),
    # Rest site: choose between resting or upgrading
    FSM.REST_SITE: ActionRoute(
        action_types=(ActionType.REST_SITE_REST, ActionType.REST_SITE_UPGRADE),
    ),
    # Map: forced to select a node
    FSM.MAP: ActionRoute(
        action_types=(ActionType.MAP_NODE_SELECT,),
    ),
}


def get_secondary_head_type(fsm: FSM, action_type: ActionType) -> HeadType | None:
    """
    Returns the secondary head to invoke, or None if terminal action.

    Args:
        fsm: Current game FSM state
        action_type: The selected action type

    Returns:
        HeadType for the secondary head, or None if no secondary head needed
    """
    match action_type:
        # Terminal actions (no secondary head)
        case ActionType.COMBAT_TURN_END | ActionType.CARD_REWARD_SKIP | ActionType.REST_SITE_REST:
            return None

        # Card in hand - depends on FSM context
        case ActionType.COMBAT_CARD_IN_HAND_SELECT:
            match fsm:
                case FSM.COMBAT_DEFAULT:
                    return HeadType.CARD_PLAY
                case FSM.COMBAT_AWAIT_TARGET_DISCARD:
                    return HeadType.CARD_DISCARD
                case _:
                    raise ValueError(f"Unexpected FSM {fsm} for {action_type}")

        # Other action types map 1:1
        case ActionType.COMBAT_MONSTER_SELECT:
            return HeadType.MONSTER_SELECT
        case ActionType.CARD_REWARD_SELECT:
            return HeadType.CARD_REWARD_SELECT
        case ActionType.REST_SITE_UPGRADE:
            return HeadType.CARD_UPGRADE
        case ActionType.MAP_NODE_SELECT:
            return HeadType.MAP_SELECT
