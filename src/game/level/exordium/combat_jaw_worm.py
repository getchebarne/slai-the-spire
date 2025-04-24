from src.game.engine.state import GameState
from src.game.entity.manager import create_entity
from src.game.factory.lib import FACTORY_LIB_MONSTER


def set_level_exoridium_combat_jaw_worm(game_state: GameState) -> None:
    game_state.entity_manager.id_monsters = [
        create_entity(
            game_state.entity_manager, FACTORY_LIB_MONSTER["Jaw Worm"](game_state.ascension_level)
        )
    ]

    return
