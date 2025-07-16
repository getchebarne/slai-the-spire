from collections import deque

from src.game.core.effect import Effect
from src.game.core.effect import EffectSelectionType
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.manager import EntityManager
from src.game.entity.manager import add_entity
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
    entity_manager = EntityManager(dict())

    # Create entities
    character, deck_starter = FACTORY_LIB_CHARACTER["Silent"](ascension_level)
    energy = create_energy(3, 3)

    # Assign corresponding ids
    entity_manager.id_character = add_entity(entity_manager, character)
    entity_manager.id_cards_in_deck = [add_entity(entity_manager, card) for card in deck_starter]
    entity_manager.id_energy = add_entity(entity_manager, energy)

    # Create map TODO: improve
    map_ = generate_map()
    for y, row in enumerate(map_):
        for x, node in enumerate(row):
            if node is None:
                continue

            id_map_node = add_entity(entity_manager, node)
            map_[y][x] = id_map_node

    entity_manager.id_map_nodes = map_
    entity_manager.id_map_node_boss = add_entity(
        entity_manager, EntityMapNode(-1, -1, RoomType.COMBAT_BOSS)
    )

    # Create effect queue
    effect_queue = deque()
    effect_queue.append(
        Effect(
            EffectType.MAP_NODE_ACTIVE_SET,
            target_type=EffectTargetType.MAP_NODE,
            selection_type=EffectSelectionType.INPUT,
        ),
    )

    return GameState(ascension_level, entity_manager, effect_queue, None)
