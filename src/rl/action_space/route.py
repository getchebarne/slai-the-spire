from src.rl.action_space.types import HeadTypePrimary
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


def get_route_primary(view_game_states: list[ViewGameState]) -> dict[HeadTypePrimary, list[int]]:
    route = {head_type_primary: [] for head_type_primary in HeadTypePrimary}

    for i, view_game_state in enumerate(view_game_states):
        head_type_primary = _VIEW_FSM_TO_HEAD_TYPE_PRIMARY[view_game_state.fsm]
        route[head_type_primary].append(i)

    return route
