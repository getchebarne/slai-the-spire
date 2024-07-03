from src.game.combat.state import Card
from src.game.combat.state import Effect
from src.game.combat.state import EffectTargetType
from src.game.combat.state import GameState


def add_effects_to_bot(context: GameState, *effects: Effect) -> None:
    context.effect_queue.extend(effects)


def add_effects_to_top(context: GameState, *effects: Effect) -> None:
    context.effect_queue.extendleft(effects[::-1])


def card_requires_target(card: Card) -> bool:
    for effect in card.effects:
        if effect.target_type == EffectTargetType.CARD_TARGET:
            return True

    return False
