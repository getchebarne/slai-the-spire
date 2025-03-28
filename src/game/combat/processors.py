import random
from dataclasses import dataclass

from src.game.combat.ai import ais
from src.game.combat.entities import Card
from src.game.combat.entities import Effect
from src.game.combat.entities import EffectType
from src.game.combat.entities import EntityManager


WEAK_FACTOR = 0.75
BLOCK_MAX = 999


@dataclass(frozen=True)
class ToBeQueuedEffect:
    effect: Effect
    id_source: int | None = None
    id_target: int | None = None


def apply_effect(
    entity_manager: EntityManager,
    effect_type: EffectType,
    effect_value: int | None,
    id_source: int | None,
    id_target: int | None,
) -> tuple[list[ToBeQueuedEffect], list[ToBeQueuedEffect]]:
    if effect_type == EffectType.DEAL_DAMAGE:
        return _apply_deal_damage(entity_manager, id_source, id_target, effect_value)

    if effect_type == EffectType.GAIN_BLOCK:
        return _apply_gain_block(entity_manager, id_target, effect_value)

    if effect_type == EffectType.PLAY_CARD:
        return _apply_play_card(entity_manager, id_target)

    if effect_type == EffectType.DRAW_CARD:
        return _apply_draw_card(entity_manager, effect_value)

    if effect_type == EffectType.REFILL_ENERGY:
        return _apply_refill_energy(entity_manager)

    if effect_type == EffectType.DISCARD:
        return _apply_discard(entity_manager, id_target)

    if effect_type == EffectType.ZERO_BLOCK:
        return _apply_zero_block(entity_manager, id_target)

    if effect_type == EffectType.DECREASE_ENERGY:
        return _apply_decrease_energy(entity_manager, effect_value)

    if effect_type == EffectType.SHUFFLE_DECK_INTO_DRAW_PILE:
        return _apply_shuffle_deck_into_draw_pile(entity_manager)

    if effect_type == EffectType.UPDATE_MOVE:
        return _apply_update_move(entity_manager, id_target)

    raise ValueError(f"Unsupported effect type: {effect_type}")


# TODO: rename to damage
def _apply_deal_damage(
    entity_manager: EntityManager, id_source: int, id_target: int, value: int
) -> tuple[list[tuple[Effect, int, int]], list[tuple[Effect, int, int]]]:
    source = entity_manager.entities[id_source]

    # TODO: think if there's a better solution
    if isinstance(source, Card):
        source = entity_manager.entities[entity_manager.id_character]

    target = entity_manager.entities[id_target]

    # Apply strength
    value += source.modifier_strength.stacks_current

    # Apply weak
    if source.modifier_weak.stacks_current > 0:
        value *= WEAK_FACTOR

    # Calculate damage over block
    value = int(value)
    damage_over_block = max(0, value - target.block_current)

    # Apply changes
    target.block_current = max(0, target.block_current - value)
    target.health_current = max(0, target.health_current - damage_over_block)

    return [], []


def _apply_gain_block(
    entity_manager: EntityManager, id_target: int, value: int
) -> tuple[list[tuple[Effect, int, int]], list[tuple[Effect, int, int]]]:
    target = entity_manager.entities[id_target]

    target.block_current = min(target.block_current + value, BLOCK_MAX)
    return [], []


def _apply_play_card(
    entity_manager: EntityManager, id_target: int
) -> tuple[list[tuple[Effect, int, int]], list[tuple[Effect, int, int]]]:
    target = entity_manager.entities[id_target]

    return (
        [
            # TODO; move to engine?
            ToBeQueuedEffect(Effect(EffectType.DECREASE_ENERGY, value=target.cost)),
            ToBeQueuedEffect(Effect(EffectType.DISCARD), id_target=id_target),
            *[ToBeQueuedEffect(effect, id_source=id_target) for effect in target.effects],
        ],
        [],
    )


# TODO: handle infinite loop
def _apply_draw_card(
    entity_manager: EntityManager, amount: int
) -> tuple[list[tuple[Effect, int, int]], list[tuple[Effect, int, int]]]:
    id_cards_in_draw_pile = entity_manager.id_cards_in_draw_pile
    id_cards_in_hand = entity_manager.id_cards_in_hand
    id_cards_in_disc_pile = entity_manager.id_cards_in_disc_pile

    for _ in range(amount):
        if len(id_cards_in_draw_pile) == 0:
            # Shuffle discard pile into draw pile TODO: make effect
            id_cards_in_draw_pile.extend(id_cards_in_disc_pile)
            random.shuffle(id_cards_in_draw_pile)

            # Clear the discard pile
            id_cards_in_disc_pile.clear()

        # Draw a card from the draw pile and add to hand
        id_cards_in_hand.append(id_cards_in_draw_pile.pop(0))

    return [], []


def _apply_refill_energy(
    entity_manager: EntityManager,
) -> tuple[list[tuple[Effect, int, int]], list[tuple[Effect, int, int]]]:
    energy = entity_manager.entities[entity_manager.id_energy]
    energy.current = energy.max

    return [], []


def _apply_decrease_energy(
    entity_manager: EntityManager, value: int
) -> tuple[list[tuple[Effect, int, int]], list[tuple[Effect, int, int]]]:
    energy = entity_manager.entities[entity_manager.id_energy]

    if energy.current < value:
        raise ValueError(f"Can't dercrease current energy ({energy.current}) by {value}")

    energy.current = energy.current - value

    return [], []


def _apply_discard(
    entity_manager: EntityManager, id_target: int
) -> tuple[list[tuple[Effect, int, int]], list[tuple[Effect, int, int]]]:
    entity_manager.id_cards_in_hand.remove(id_target)
    entity_manager.id_cards_in_disc_pile.append(id_target)

    return [], []


def _apply_zero_block(
    entity_manager: EntityManager, id_target: int
) -> tuple[list[tuple[Effect, int, int]], list[tuple[Effect, int, int]]]:
    target = entity_manager.entities[id_target]

    target.block_current = 0

    return [], []


def _apply_shuffle_deck_into_draw_pile(
    entity_manager: EntityManager,
) -> tuple[list[tuple[Effect, int, int]], list[tuple[Effect, int, int]]]:
    entity_manager.id_cards_in_draw_pile = entity_manager.id_cards_in_deck.copy()
    random.shuffle(entity_manager.id_cards_in_draw_pile)

    return [], []


def _apply_update_move(
    entity_manager: EntityManager, id_target: int
) -> tuple[list[tuple[Effect, int, int]], list[tuple[Effect, int, int]]]:
    target = entity_manager.entities[id_target]

    move_new = ais[target.name](target.move_current, target.move_history)
    target.move_current = move_new
    target.move_history.append(move_new)

    return [], []
