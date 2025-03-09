import random
from dataclasses import replace

from src.game.combat.entities import Effect
from src.game.combat.entities import EffectType
from src.game.combat.entities import EffectTargetType
from src.game.combat.entities import Entities
from src.game.combat.state import QueuedEffect


WEAK_FACTOR = 0.75
BLOCK_MAX = 999


def process_effect(
    entities: Entities, source_id: int, target_id: int, effect: Effect
) -> tuple[Entities, list[QueuedEffect], list[QueuedEffect]]:
    if effect.type == EffectType.DEAL_DAMAGE:
        return _processor_deal_damage(entities, source_id, target_id, effect.value)

    if effect.type == EffectType.GAIN_BLOCK:
        return _processor_gain_block(entities, target_id, effect.value)

    if effect.type == EffectType.PLAY_CARD:
        return _processor_play_card(entities, target_id)

    if effect.type == EffectType.DRAW_CARD:
        return _processor_draw_card(entities, effect.value)

    if effect.type == EffectType.REFILL_ENERGY:
        return _processor_refill_energy(entities)

    if effect.type == EffectType.DISCARD:
        return _processor_discard(entities, target_id)

    if effect.type == EffectType.ZERO_BLOCK:
        return _processor_zero_block(entities, target_id)

    if effect.type == EffectType.DECREASE_ENERGY:
        return _processor_decrease_energy(entities, effect.value)

    raise ValueError(f"Unsupported effect type: {effect.type}")


# TODO: rename to damage
def _processor_deal_damage(
    entities: Entities, source_id: int, target_id: int, value: int
) -> tuple[Entities, list[QueuedEffect], list[QueuedEffect]]:
    source = entities.all[source_id]
    target = entities.all[target_id]

    # Apply strength
    value += source.modifier_strength.stacks_current

    # Apply weak
    if source.modifier_weak.stacks_current > 0:
        value *= WEAK_FACTOR

    # Calculate damage over block
    value = int(value)
    damage_over_block = max(0, value - target.block_current)

    # TODO; add comment
    all_new = entities.all.copy()
    all_new[target_id] = replace(
        target,
        block_current=max(0, target.block_current - value),
        health_current=max(0, target.health_current - damage_over_block),
    )

    return replace(entities, all=all_new), [], []


def _processor_gain_block(
    entities: Entities, target_id: int, value: int
) -> tuple[Entities, list[QueuedEffect], list[QueuedEffect]]:
    target = entities.all[target_id]

    # TODO; add comment
    all_new = entities.all.copy()
    all_new[target_id] = replace(
        target,
        block_current=min(target.block_current + value, BLOCK_MAX),
    )

    entities_new = replace(entities, all=all_new)

    return entities_new, [], []


def _processor_play_card(
    entities: Entities, target_id: int
) -> tuple[Entities, list[QueuedEffect], list[QueuedEffect]]:
    target = entities.all[target_id]

    # TODO: think about having source_id=character_id here
    effect_card = [
        QueuedEffect(effect, source_id=entities.character_id) for effect in target.effects
    ]

    return (
        entities,
        [
            QueuedEffect(Effect(EffectType.DECREASE_ENERGY, value=target.cost)),
            # TODO: add support for queueing effects w/ target_id
            QueuedEffect(Effect(EffectType.DISCARD, target_type=EffectTargetType.CARD_ACTIVE)),
            *effect_card,
        ],
        [],
    )


# TODO: handle infinite loop
def _processor_draw_card(
    entities: Entities, amount: int
) -> tuple[Entities, list[QueuedEffect], list[QueuedEffect]]:
    card_in_draw_pile_ids = entities.card_in_draw_pile_ids.copy()
    card_in_discard_pile_ids = entities.card_in_discard_pile_ids.copy()
    card_in_hand_ids = entities.card_in_hand_ids.copy()

    for _ in range(amount):
        if len(card_in_draw_pile_ids) == 0:
            # Shuffle discard pile into draw pile
            # TODO: make effect
            card_in_draw_pile_ids = list(card_in_discard_pile_ids)
            random.shuffle(card_in_draw_pile_ids)

            card_in_discard_pile_ids = set()

        card_in_hand_ids.append(card_in_draw_pile_ids.pop(0))

    entities_new = replace(
        entities,
        card_in_draw_pile_ids=card_in_draw_pile_ids,
        card_in_hand_ids=card_in_hand_ids,
        card_in_discard_pile_ids=card_in_discard_pile_ids,
    )

    return entities_new, [], []


def _processor_refill_energy(
    entities: Entities,
) -> tuple[list[tuple[int, Effect]], list[tuple[int, Effect]]]:
    energy_id = entities.energy_id
    energy = entities.all[energy_id]

    # Create a new energy entity with updated current value
    energy_new = replace(energy, current=energy.max)

    # Create a new list with the updated entity
    all_new = entities.all.copy()
    all_new[energy_id] = energy_new

    # Create new Entities with the updated list
    entities_new = replace(entities, all=all_new)

    return entities_new, [], []


def _processor_decrease_energy(
    entities: Entities, value: int
) -> tuple[list[tuple[int, Effect]], list[tuple[int, Effect]]]:
    energy_id = entities.energy_id
    energy = entities.all[energy_id]

    # Create a new energy entity with updated current value
    energy_new = replace(energy, current=energy.current - value)

    # Create a new list with the updated entity
    all_new = entities.all.copy()
    all_new[energy_id] = energy_new

    # Create new Entities with the updated list
    entities_new = replace(entities, all=all_new)

    return entities_new, [], []


def _processor_discard(
    entities: Entities, target_id: int
) -> tuple[list[tuple[int, Effect]], list[tuple[int, Effect]]]:
    # Create copies of collections we need to modify
    card_in_hand_ids = entities.card_in_hand_ids.copy()
    card_in_discard_pile_ids = entities.card_in_discard_pile_ids.copy()

    # Update collections
    card_in_hand_ids.remove(target_id)
    card_in_discard_pile_ids.add(target_id)

    # Create new Entities with updated collections
    new_entities = replace(
        entities,
        card_in_hand_ids=card_in_hand_ids,
        card_in_discard_pile_ids=card_in_discard_pile_ids,
    )

    return new_entities, [], []


def _processor_zero_block(
    entities: Entities, target_id: int
) -> tuple[list[tuple[int, Effect]], list[tuple[int, Effect]]]:
    target = entities.all[target_id]

    # Create a new target entity with block set to 0
    target_new = replace(target, block_current=0)

    # Create a new list with the updated entity
    all_new = entities.all.copy()
    all_new[target_id] = target_new

    # Create new Entities with the updated list
    new_entities = replace(entities, all=all_new)

    return new_entities, [], []
