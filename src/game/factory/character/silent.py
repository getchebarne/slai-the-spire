from src.game.entity.card import EntityCard
from src.game.entity.character import EntityCharacter
from src.game.factory.card.defend import create_card_defend
from src.game.factory.card.neutralize import create_card_neutralize
from src.game.factory.card.strike import create_card_strike
from src.game.factory.card.survivor import create_card_survivor
from src.game.factory.lib import register_factory
from src.game.types import AscensionLevel


_NAME = "Silent"
_HEALTH_MAX = 70
_FACTOR_HEALTH_CURRENT_ASC_6 = 0.90
_PENALTY_HEALTH_MAX_ASC_14 = 4


@register_factory(_NAME)
def create_character_silent(
    ascension_level: AscensionLevel,
) -> tuple[EntityCharacter, list[EntityCard]]:
    character = _create_character(ascension_level)
    deck_starter = _create_starter_deck()

    return character, deck_starter


def _create_character(ascension_level: AscensionLevel) -> EntityCharacter:
    health_max = _HEALTH_MAX
    health_current = health_max
    if ascension_level >= 6:
        if ascension_level >= 14:
            health_max -= _PENALTY_HEALTH_MAX_ASC_14
            health_current = health_max

        health_current = int(_FACTOR_HEALTH_CURRENT_ASC_6 * health_current)

    return EntityCharacter("Silent", health_current=health_current, health_max=health_max)


def _create_starter_deck() -> list[EntityCard]:
    return [
        # Strikes
        create_card_strike(upgraded=False),
        create_card_strike(upgraded=False),
        create_card_strike(upgraded=False),
        create_card_strike(upgraded=False),
        create_card_strike(upgraded=False),
        # Defends
        create_card_defend(upgraded=False),
        create_card_defend(upgraded=False),
        create_card_defend(upgraded=False),
        create_card_defend(upgraded=False),
        create_card_defend(upgraded=False),
        # Survivor
        create_card_survivor(upgraded=False),
        # Neutralize
        create_card_neutralize(upgraded=False),
    ]
