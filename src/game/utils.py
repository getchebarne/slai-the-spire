from src.game.core.effect import EffectTargetType
from src.game.entity.actor import ModifierType
from src.game.entity.card import EntityCard
from src.game.entity.character import EntityCharacter
from src.game.entity.manager import EntityManager
from src.game.entity.monster import EntityMonster


def is_character_dead(entity_manager: EntityManager) -> bool:
    return entity_manager.character.health_current <= 0


def is_combat_over(entity_manager: EntityManager) -> bool:
    return is_character_dead(entity_manager) or (not entity_manager.id_monsters)


def does_card_require_target(card: EntityCard) -> bool:
    for effect in card.effects:
        if effect.target_type == EffectTargetType.CARD_TARGET:
            return True

    return False


def get_corrected_intent_damage(
    damage: int, monster: EntityMonster, character: EntityCharacter
) -> int:
    if ModifierType.STRENGTH in monster.modifier_map:
        damage += monster.modifier_map[ModifierType.STRENGTH].stacks_current

    if ModifierType.WEAK in monster.modifier_map:
        damage *= 0.75  # TODO: same as processor

    if ModifierType.VULNERABLE in character.modifier_map:
        damage *= 1.50  # TODO: same as processor

    return int(damage)
