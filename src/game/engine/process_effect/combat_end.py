from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.entity.manager import EntityManager
from src.game.entity.map_node import RoomType


# TODO: will need to add end of combat triggers, such as "Blood Vial"
def process_effect_combat_end(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    # Clear hand, draw pile, discard pile, and exhaust pile
    entity_manager.id_cards_in_hand = []
    entity_manager.id_cards_in_draw_pile = []
    entity_manager.id_cards_in_disc_pile = []
    entity_manager.id_cards_in_exhaust_pile = []

    # Clear character's modifiers
    character = entity_manager.entities[entity_manager.id_character]
    character.modifier_map = dict()

    # Get current room type
    map_node_active = entity_manager.entities[entity_manager.id_map_node_active]
    if map_node_active.room_type == RoomType.COMBAT_BOSS:
        # Game end
        return [], [Effect(EffectType.GAME_END)]

    if map_node_active.room_type == RoomType.COMBAT_MONSTER:
        return [Effect(EffectType.CARD_REWARD_ROLL)], []

    raise ValueError(f"Unsupported room type: {map_node_active.room_type}")
