from src.rl.action_space.types import HeadTypePrimary
from src.rl.action_space.types import NUM_PRIMARY_HEADS
from src.game.view.fsm import ViewFSM
from src.game.view.state import ViewGameState


_VIEW_FSM_TO_HEAD_TYPE_PRIMARY = {
    ViewFSM.CARD_REWARD: HeadTypePrimary.CARD_REWARD,
    ViewFSM.COMBAT_DEFAULT: HeadTypePrimary.COMBAT_DEFAULT,
    ViewFSM.COMBAT_AWAIT_TARGET_CARD: HeadTypePrimary.COMBAT_MONSTER_SELECT,
    ViewFSM.COMBAT_AWAIT_TARGET_DISCARD: HeadTypePrimary.COMBAT_CARD_DISCARD,
    ViewFSM.MAP: HeadTypePrimary.MAP_SELECT,
    ViewFSM.REST_SITE: HeadTypePrimary.REST_SITE,
}


def get_route_primary(view_game_states: list[ViewGameState]) -> list[list[int]]:
    """Route game states to primary head groups. Returns list indexed by int(HeadTypePrimary)."""
    route: list[list[int]] = [[] for _ in range(NUM_PRIMARY_HEADS)]

    for i, view_game_state in enumerate(view_game_states):
        htp = _VIEW_FSM_TO_HEAD_TYPE_PRIMARY[view_game_state.fsm]
        route[htp].append(i)

    return route
