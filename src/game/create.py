from collections import deque

from src.game.const import MAP_HEIGHT
from src.game.entity.manager import EntityManager
from src.game.entity.map_node import EntityMapNode
from src.game.entity.map_node import RoomType
from src.game.factory.energy import create_energy
from src.game.factory.lib import FACTORY_LIB_CHARACTER
from src.game.map_ import generate_map
from src.game.state import GameState
from src.game.types_ import AscensionLevel


# TODO: parametrize deck, monster, etc.
def create_game_state(ascension_level: AscensionLevel) -> GameState:
    # Create empty EntityManager
    entity_manager = EntityManager()

    # Create entities
    character, deck_starter = FACTORY_LIB_CHARACTER["Silent"](ascension_level)
    energy = create_energy(3, 3)

    # Assign directly
    entity_manager.character = character
    entity_manager.deck = deck_starter
    entity_manager.energy = energy

    # Create map TODO: improve
    entity_manager.map_nodes = generate_map()
    entity_manager.map_node_boss = EntityMapNode(MAP_HEIGHT, -1, RoomType.COMBAT_BOSS)

    # Create effect queue
    effect_queue = deque()

    return GameState(ascension_level, entity_manager, effect_queue, None)
