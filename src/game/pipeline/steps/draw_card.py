import random

from src.game.context import Context
from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.pipeline.steps.base import BaseStep


def _draw_one_card(context: Context) -> None:
    # If the draw pile is empty, shuffle the discard pile and send it to the draw pile
    if len(context.draw_pile) == 0:
        random.shuffle(context.disc_pile)
        context.draw_pile = context.disc_pile
        context.disc_pile = []

    # Draw one card from draw pile
    context.hand.append(context.draw_pile.pop(0))


class DrawCard(BaseStep):
    def _apply_effect(self, context: Context, effect: Effect) -> None:
        num_cards_to_draw = effect.value
        for _ in range(num_cards_to_draw):
            _draw_one_card(context)

    def _condition(self, context: Context, effect: Effect) -> bool:
        return effect.type == EffectType.DRAW_CARD
