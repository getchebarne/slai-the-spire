import random
from typing import Callable

from src.game.combat.factories import weak
from src.game.combat.state import Card
from src.game.combat.state import Effect
from src.game.combat.state import EffectTargetType
from src.game.combat.state import EffectType
from src.game.combat.state import GameState
from src.game.combat.state import ModifierType


def _processor_deal_damage(state: GameState, **kwargs) -> None:
    target = state.get_entity(kwargs["target_id"])
    damage = int(kwargs["value"])

    health = target.health
    block = target.block

    # Remove block
    damage_over_block = max(0, damage - block.current)
    block.current = max(0, block.current - damage)

    # Remove health
    health.current = max(0, health.current - damage_over_block)


def _processor_gain_block(state: GameState, **kwargs) -> None:
    target = state.get_entity(kwargs["target_id"])
    value = int(kwargs["value"])

    block = target.block
    block.current = min(block.current + value, block.max)


def _processor_play_card(state: GameState, **kwargs) -> None:
    target = state.get_entity(kwargs["target_id"])
    energy = state.get_entity(state.energy_id)

    if target.cost > energy.current:
        raise ValueError(f"Can't play card {target} with {energy.current} energy")

    # Subtract energy
    # TODO: should be effect
    energy.current -= target.cost

    # Add effect to discard the card
    kwargs["add_to_bot_callback"](
        None, Effect(EffectType.DISCARD, target_type=EffectTargetType.CARD_ACTIVE)
    )

    # Add card's effects to pipeline
    # TODO: think about creating effects w/ target already
    kwargs["add_to_bot_callback"](kwargs["target_id"], *target.effects)


def _processor_gain_weak(state: GameState, **kwargs) -> None:
    target = state.get_entity(kwargs["target_id"])
    value = int(kwargs["value"])

    # TODO: use defaultdict?
    try:
        modifier_weak = target.modifiers[ModifierType.WEAK]
        modifier_weak.stacks = min(modifier_weak.stacks + value, modifier_weak.stacks_max)

    except KeyError:
        modifier_weak = weak()
        modifier_weak.stacks = value
        target.modifiers[ModifierType.WEAK] = modifier_weak


def _processor_apply_weak(state: GameState, **kwargs) -> None:
    source = state.get_entity(kwargs["source_id"])
    value = kwargs["value"]

    # TODO: improve
    if isinstance(source, Card):
        source = state.get_entity(state.character_id)

    if ModifierType.WEAK in source.modifiers:
        value *= 0.75


# TODO: handle infinite loop
def _processor_draw_card(state: GameState, **kwargs) -> None:
    value = int(kwargs["value"])

    for _ in range(value):
        if len(state.card_in_draw_pile_ids) == 0:
            # Shuffle discard pile into draw pile
            # TODO: make effect
            state.card_in_draw_pile_ids = list(state.card_in_discard_pile_ids)
            random.shuffle(state.card_in_draw_pile_ids)

            state.card_in_discard_pile_ids = set()

        state.card_in_hand_ids.append(state.card_in_draw_pile_ids.pop(0))


def _processor_refill_energy(state: GameState, **kwargs) -> None:
    energy = state.get_entity(state.energy_id)
    energy.current = energy.max


def _processor_discard(state: GameState, **kwargs) -> None:
    state.card_in_hand_ids.remove(kwargs["target_id"])
    state.card_in_discard_pile_ids.add(kwargs["target_id"])


def _processor_zero_block(state: GameState, **kwargs) -> None:
    target = state.get_entity(kwargs["target_id"])

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
    EffectType.PLAY_CARD: [_processor_play_card],
}
