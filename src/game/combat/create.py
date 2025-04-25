from collections import deque

from src.game.engine.state import GameState
from src.game.entity.manager import EntityManager
from src.game.entity.manager import create_entity
from src.game.factory.energy import create_energy
from src.game.factory.lib import FACTORY_LIB_CHARACTER
from src.game.types_ import AscensionLevel
from src.game.types_ import RoomType


# TODO: parametrize deck, monster, etc.
def create_game_state(ascension_level: AscensionLevel) -> GameState:
    # Create empty EntityManager
    entity_manager = EntityManager([])

    # Create entities
    character, deck_starter = FACTORY_LIB_CHARACTER["Silent"](ascension_level)
    energy = create_energy(3, 3)

    # Assign corresponding ids
    entity_manager.id_character = create_entity(entity_manager, character)
    entity_manager.id_cards_in_deck = [
        create_entity(entity_manager, card) for card in deck_starter
    ]
    entity_manager.id_energy = create_entity(entity_manager, energy)

    # Create map
    map_ = [
        RoomType.COMBAT_MONSTER,
        RoomType.REST_SITE,
        RoomType.COMBAT_MONSTER,
    ]

    # Create effect queue
    effect_queue = deque()

    return GameState(ascension_level, entity_manager, effect_queue, None, map_)
