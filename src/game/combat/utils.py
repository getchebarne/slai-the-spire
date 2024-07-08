from src.game.combat.entities import Card
from src.game.combat.entities import EffectTargetType
from src.game.combat.entities import Entities


def card_requires_target(card: Card) -> bool:
    for effect in card.effects:
        if effect.target_type == EffectTargetType.CARD_TARGET:
            return True

    return False


def is_game_over(entities: Entities) -> bool:
    return entities.get_character().health.current <= 0 or all(
        [monster.health.current <= 0 for monster in entities.get_monsters()]
    )
