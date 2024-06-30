from typing import Optional

from src.game.combat.context import Card
from src.game.combat.context import Effect
from src.game.combat.context import EffectTargetType
from src.game.combat.context import GameContext


def add_effects_to_bot(context: GameContext, *effects: Effect) -> None:
    context.effect_queue.extend(effects)


def add_effects_to_top(context: GameContext, *effects: Effect) -> None:
    context.effect_queue.extendleft(effects[::-1])


def card_requires_target(card: Card) -> bool:
    for effect in card.effects:
        if effect.target_type == EffectTargetType.CARD_TARGET:
            return True

    return False


def get_active_card(context: GameContext) -> Optional[Card]:
    for card in context.hand:
        if card.is_active:
            return card
