from typing import TypeVar

from src.game.core.effect import EffectSelectionType
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.card import EntityCard
from src.game.entity.manager import EntityManager


T = TypeVar("T")


def remove_by_identity(lst: list[T], item: T) -> bool:
    for i, x in enumerate(lst):
        if x is item:
            lst.pop(i)
            return True

    return False


def is_character_dead(entity_manager: EntityManager) -> bool:
    return entity_manager.character.health_current <= 0


def is_combat_over(entity_manager: EntityManager) -> bool:
    return is_character_dead(entity_manager) or (not entity_manager.monsters)


def does_card_require_target(card: EntityCard) -> bool:
    return any(effect.target_type == EffectTargetType.CARD_TARGET for effect in card.effects)


# TODO: add number of cards to discard
def does_card_require_discard(card: EntityCard) -> bool:
    return any(
        (
            effect.type == EffectType.CARD_DISCARD
            and effect.selection_type == EffectSelectionType.INPUT
        )
        for effect in card.effects
    )
