import random
from typing import Callable

from src.game.combat.entities import Card
from src.game.combat.entities import Effect
from src.game.combat.entities import EffectTargetType
from src.game.combat.entities import EffectType
from src.game.combat.entities import Entities
from src.game.combat.entities import ModifierType
from src.game.combat.factories import weak


def _processor_deal_damage(
    entities: Entities, **kwargs
) -> tuple[list[tuple[int, Effect]], list[tuple[int, Effect]]]:
    target = entities.get_entity(kwargs["target_id"])
    damage = int(kwargs["value"])

    health = target.health
    block = target.block

    # Remove block
    damage_over_block = max(0, damage - block.current)
    block.current = max(0, block.current - damage)

    # Remove health
    health.current = max(0, health.current - damage_over_block)

    return [], []


def _processor_gain_block(
    entities: Entities, **kwargs
) -> tuple[list[tuple[int, Effect]], list[tuple[int, Effect]]]:
    target = entities.get_entity(kwargs["target_id"])
    value = int(kwargs["value"])

    block = target.block
    block.current = min(block.current + value, block.max)

    return [], []


def _processor_play_card(
    entities: Entities, **kwargs
) -> tuple[list[tuple[int, Effect]], list[tuple[int, Effect]]]:
    target_id = kwargs["target_id"]
    target = entities.get_entity(target_id)
    energy = entities.get_entity(entities.energy_id)

    if target.cost > energy.current:
        raise ValueError(f"Can't play card {target} with {energy.current} energy")

    # Subtract energy
    # TODO: should be effect
    energy.current -= target.cost

    # Add effect to discard the card
    # TODO: think about creating effects w/ target already
    effect_discard = (None, Effect(EffectType.DISCARD, target_type=EffectTargetType.CARD_ACTIVE))
    effect_card = [(target_id, effect) for effect in target.effects]

    return [effect_discard] + effect_card, []


def _processor_gain_weak(
    entities: Entities, **kwargs
) -> tuple[list[tuple[int, Effect]], list[tuple[int, Effect]]]:
    target = entities.get_entity(kwargs["target_id"])
    value = int(kwargs["value"])

    # TODO: use defaultdict?
    try:
        modifier_weak = target.modifiers[ModifierType.WEAK]
        modifier_weak.stacks = min(modifier_weak.stacks + value, modifier_weak.stacks_max)

    except KeyError:
        modifier_weak = weak()
        modifier_weak.stacks = value
        target.modifiers[ModifierType.WEAK] = modifier_weak

    return [], []


def _processor_apply_weak(
    entities: Entities, **kwargs
) -> tuple[list[tuple[int, Effect]], list[tuple[int, Effect]]]:
    source = entities.get_entity(kwargs["source_id"])
    value = kwargs["value"]

    # TODO: improve
    if isinstance(source, Card):
        source = entities.get_entity(entities.character_id)

    if ModifierType.WEAK in source.modifiers:
        value *= 0.75

    return [], []


# TODO: handle infinite loop
def _processor_draw_card(
    entities: Entities, **kwargs
) -> tuple[list[tuple[int, Effect]], list[tuple[int, Effect]]]:
    value = int(kwargs["value"])

    for _ in range(value):
        if len(entities.card_in_draw_pile_ids) == 0:
            # Shuffle discard pile into draw pile
            # TODO: make effect
            entities.card_in_draw_pile_ids = list(entities.card_in_discard_pile_ids)
            random.shuffle(entities.card_in_draw_pile_ids)

            entities.card_in_discard_pile_ids = set()

        entities.card_in_hand_ids.append(entities.card_in_draw_pile_ids.pop(0))

    return [], []


def _processor_refill_energy(
    entities: Entities, **kwargs
) -> tuple[list[tuple[int, Effect]], list[tuple[int, Effect]]]:
    energy = entities.get_entity(entities.energy_id)
    energy.current = energy.max

    return [], []


def _processor_discard(
    entities: Entities, **kwargs
) -> tuple[list[tuple[int, Effect]], list[tuple[int, Effect]]]:
    entities.card_in_hand_ids.remove(kwargs["target_id"])
    entities.card_in_discard_pile_ids.add(kwargs["target_id"])

    return [], []


def _processor_zero_block(
    entities: Entities, **kwargs
) -> tuple[list[tuple[int, Effect]], list[tuple[int, Effect]]]:
    target = entities.get_entity(kwargs["target_id"])

    target.block.current = 0

    return [], []


def get_effect_processors(effect_type: EffectType) -> list[Callable]:  # TODO: add argument type
    return processors[effect_type]

    return [], []


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
