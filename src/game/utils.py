from src.game.core.effect import EffectTargetType
from src.game.entity.card import EntityCard
from src.game.entity.manager import EntityManager


def is_character_dead(entity_manager: EntityManager) -> bool:
    character = entity_manager.entities[entity_manager.id_character]

    return character.health_current <= 0


def is_combat_over(entity_manager: EntityManager) -> bool:
    return is_character_dead(entity_manager) or (not entity_manager.id_monsters)


def does_card_require_target(card: EntityCard) -> bool:
    for effect in card.effects:
        if effect.target_type == EffectTargetType.CARD_TARGET:
            return True

    return False
