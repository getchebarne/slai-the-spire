import random
from typing import Callable

from src.game.combat.factories import weak
from src.game.combat.state import Card
from src.game.combat.state import EffectType
from src.game.combat.state import GameState
from src.game.combat.state import ModifierType


def _processor_deal_damage(state: GameState) -> None:
    target = state.get_effect_target()
    value = int(state.effect_value)

    health = target.health
    block = target.block

    # Remove block
    damage_over_block = max(0, value - block.current)
    block.current = max(0, block.current - value)

    # Remove health
    health.current = max(0, health.current - damage_over_block)


def _processor_gain_block(state: GameState) -> None:
    target = state.get_effect_target()
    value = state.effect_value

    block = target.block
    block.current = min(block.current + value, block.max)


def _processor_gain_weak(state: GameState) -> None:
    target = state.get_effect_target()
    value = state.effect_value

    try:
        modifier_weak = target.modifiers[ModifierType.WEAK]
        modifier_weak.stacks = min(modifier_weak.stacks + value, modifier_weak.stacks_max)

    except KeyError:
        modifier_weak = weak()
        modifier_weak.stacks = value
        target.modifiers[ModifierType.WEAK] = modifier_weak


def _processor_apply_weak(state: GameState) -> None:
    source = state.get_entity(state.effect_source_id)

    # TODO: improve
    if isinstance(source, Card):
        source = state.get_character()

    value = state.effect_value

    if ModifierType.WEAK in source.modifiers:
        value *= 0.75


# TODO: handle infinite loop
def _processor_draw_card(state: GameState) -> None:
    value = state.effect_value

    for _ in range(value):
        if len(state.card_in_draw_pile_ids) == 0:
            # Shuffle discard pile into draw pile
            # TODO: make effect

            state.card_in_draw_pile_ids = list(state.card_in_discard_pile_ids)
            random.shuffle(state.card_in_draw_pile_ids)

            state.card_in_discard_pile_ids = set()

        state.card_in_hand_ids.append(state.card_in_draw_pile_ids.pop(0))


def _processor_refill_energy(state: GameState) -> None:
    energy = state.get_energy()
    energy.current = energy.max


def _processor_discard(state: GameState) -> None:
    target_id = state.effect_target_id

    state.card_in_hand_ids.remove(target_id)
    state.card_in_discard_pile_ids.add(target_id)


def _processor_zero_block(state: GameState) -> None:
    target = state.get_effect_target()

    target.block.current = 0


def get_effect_processors(effect_type: EffectType) -> list[Callable]:  # TODO: add argument type
    return processors[effect_type]


# Dictionary to hold effect type to processing function mappings
processors = {
    EffectType.DEAL_DAMAGE: [_processor_apply_weak, _processor_deal_damage],
    EffectType.GAIN_BLOCK: [_processor_gain_block],
    EffectType.GAIN_WEAK: [_processor_gain_weak],
    EffectType.DRAW_CARD: [_processor_draw_card],
    EffectType.REFILL_ENERGY: [_processor_refill_energy],
    EffectType.DISCARD: [_processor_discard],
    EffectType.ZERO_BLOCK: [_processor_zero_block],
}
