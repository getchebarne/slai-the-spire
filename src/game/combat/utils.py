from src.game.combat.state import Card
from src.game.combat.state import EffectTargetType
from src.game.combat.state import GameState


def card_requires_target(card: Card) -> bool:
    for effect in card.effects:
        if effect.target_type == EffectTargetType.CARD_TARGET:
            return True

    return False


def is_game_over(state: GameState) -> bool:
    return state.get_character().health.current <= 0 or all(
        [monster.health.current <= 0 for monster in state.get_monsters()]
    )
