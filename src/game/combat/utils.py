from src.game.combat.effect import EffectTargetType
from src.game.combat.entities import Card
from src.game.combat.entities import EntityManager


def is_game_over(entity_manager: EntityManager) -> bool:
    character = entity_manager.entities[entity_manager.id_character]
    monsters = [entity_manager.entities[id_monster] for id_monster in entity_manager.id_monsters]

    return character.health_current <= 0 or all(
        [monster.health_current <= 0 for monster in monsters]
    )


def does_card_require_target(card: Card) -> bool:
    for effect in card.effects:
        if effect.target_type == EffectTargetType.CARD_TARGET:
            return True

    return False
