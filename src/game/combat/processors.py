import random
from typing import Callable

from src.game.combat.context import EffectType
from src.game.combat.context import GameContext


def _processor_deal_damage(context: GameContext) -> None:
    target = context.effect_target
    value = context.effect_value

    health = target.health
    block = target.block

    # Remove block
    damage_over_block = max(0, value - block.current)
    block.current = max(0, block.current - value)

    # Remove health
    health.current = max(0, health.current - damage_over_block)


def _processor_gain_block(context: GameContext) -> None:
    target = context.effect_target
    value = context.effect_value

    block = target.block
    block.current = min(block.current + value, block.max)


# TODO: handle infinite loop
def _processor_draw_card(context: GameContext) -> None:
    value = context.effect_value

    for _ in range(value):
        if len(context.draw_pile) == 0:
            # Shuffle discard pile into draw pile
            # TODO: make effect
            context.draw_pile = list(context.discard_pile)
            random.shuffle(context.draw_pile)

            context.discard_pile = set()

        context.hand.append(context.draw_pile.pop(0))


def _processor_refill_energy(context: GameContext) -> None:
    context.energy.current = context.energy.max


def _processor_discard(context: GameContext) -> None:
    target = context.effect_target

    context.hand.remove(target)
    context.discard_pile.add(target)


def _processor_zero_block(context: GameContext) -> None:
    target = context.effect_target

    target.block.current = 0


def get_effect_processors(effect_type: EffectType) -> list[Callable]:  # TODO: add argument type
    return processors[effect_type]


# Dictionary to hold effect type to processing function mappings
processors = {
    EffectType.DEAL_DAMAGE: [_processor_deal_damage],
    EffectType.GAIN_BLOCK: [_processor_gain_block],
    EffectType.DRAW_CARD: [_processor_draw_card],
    EffectType.REFILL_ENERGY: [_processor_refill_energy],
    EffectType.DISCARD: [_processor_discard],
    EffectType.ZERO_BLOCK: [_processor_zero_block],
}
