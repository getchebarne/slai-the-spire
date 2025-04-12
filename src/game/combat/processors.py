import random

from src.game.combat.ai import ais
from src.game.combat.effect import Effect
from src.game.combat.effect import EffectType
from src.game.combat.effect import SourcedEffect
from src.game.combat.entities import Card
from src.game.combat.entities import EntityManager
from src.game.combat.entities import ModifierType
from src.game.combat.factories import create_modifier_strength
from src.game.combat.factories import create_modifier_weak
from src.game.combat.phase import get_end_of_turn_effects
from src.game.combat.phase import get_start_of_turn_effects


WEAK_FACTOR = 0.75
BLOCK_MAX = 999


def apply_effect(
    entity_manager: EntityManager,
    effect_type: EffectType,
    effect_value: int | None,
    id_source: int | None,
    id_target: int | None,
) -> tuple[list[SourcedEffect], list[SourcedEffect]]:
    if effect_type == EffectType.DEAL_DAMAGE:
        return _apply_deal_damage(entity_manager, id_source, id_target, effect_value)

    if effect_type == EffectType.GAIN_BLOCK:
        return _apply_gain_block(entity_manager, id_target, effect_value)

    if effect_type == EffectType.PLAY_CARD:
        return _apply_play_card(entity_manager, id_target)

    if effect_type == EffectType.DRAW_CARD:
        return _apply_draw_card(entity_manager, effect_value)

    if effect_type == EffectType.END_TURN:
        return _apply_end_turn(entity_manager)

    if effect_type == EffectType.REFILL_ENERGY:
        return _apply_refill_energy(entity_manager)

    if effect_type == EffectType.DISCARD:
        return _apply_discard(entity_manager, id_target)

    if effect_type == EffectType.ZERO_BLOCK:
        return _apply_zero_block(entity_manager, id_target)

    if effect_type == EffectType.DECREASE_ENERGY:
        return _apply_decrease_energy(entity_manager, effect_value)

    if effect_type == EffectType.GAIN_WEAK:
        return _apply_gain_weak(entity_manager, id_target, effect_value)

    if effect_type == EffectType.MOD_TICK:
        return _apply_mod_tick(entity_manager, id_target)

    if effect_type == EffectType.CARD_ACTIVE_SET:
        return _apply_card_active_set(entity_manager, id_target)

    if effect_type == EffectType.CARD_ACTIVE_CLEAR:
        return _apply_card_active_clear(entity_manager)

    if effect_type == EffectType.TARGET_EFFECT_SET:
        return _apply_target_effect_set(entity_manager, id_target)

    if effect_type == EffectType.TARGET_EFFECT_CLEAR:
        return _apply_target_effect_clear(entity_manager)

    if effect_type == EffectType.TARGET_CARD_SET:
        return _apply_target_card_set(entity_manager, id_target)

    if effect_type == EffectType.TARGET_CARD_CLEAR:
        return _apply_target_card_clear(entity_manager)

    if effect_type == EffectType.SHUFFLE_DECK_INTO_DRAW_PILE:
        return _apply_shuffle_deck_into_draw_pile(entity_manager)

    if effect_type == EffectType.UPDATE_MOVE:
        return _apply_update_move(entity_manager, id_target)

    if effect_type == EffectType.GAIN_STRENGTH:
        return _apply_gain_strength(entity_manager, id_target, effect_value)

    raise ValueError(f"Unsupported effect type: {effect_type}")


def _apply_end_turn(
    entity_manager: EntityManager,
) -> tuple[list[SourcedEffect], list[SourcedEffect]]:
    # Character's turn end
    sourced_effects = get_end_of_turn_effects(entity_manager, entity_manager.id_character)

    # Monsters
    for id_monster in entity_manager.id_monsters:
        monster = entity_manager.entities[id_monster]

        # Turn start
        sourced_effects += get_start_of_turn_effects(entity_manager, id_monster)

        # Move's effects
        sourced_effects += [
            SourcedEffect(effect, id_monster) for effect in monster.move_current.effects
        ]

        # Update move
        sourced_effects += [SourcedEffect(Effect(EffectType.UPDATE_MOVE), id_target=id_monster)]

        # Turn end
        sourced_effects += get_end_of_turn_effects(entity_manager, id_monster)

    # Character's turn start
    sourced_effects += get_start_of_turn_effects(entity_manager, entity_manager.id_character)

    return sourced_effects, []


def _apply_card_active_set(
    entity_manager: EntityManager, id_target: int
) -> tuple[list[SourcedEffect], list[SourcedEffect]]:
    entity_manager.id_card_active = id_target

    return [], []


def _apply_card_active_clear(
    entity_manager: EntityManager,
) -> tuple[list[SourcedEffect], list[SourcedEffect]]:
    entity_manager.id_card_active = None

    return [], []


def _apply_target_effect_set(
    entity_manager: EntityManager, id_target: int
) -> tuple[list[SourcedEffect], list[SourcedEffect]]:
    entity_manager.id_effect_target = id_target

    return [], []


def _apply_target_effect_clear(
    entity_manager: EntityManager,
) -> tuple[list[SourcedEffect], list[SourcedEffect]]:
    entity_manager.id_effect_target = None

    return [], []


def _apply_target_card_set(
    entity_manager: EntityManager, id_target: int
) -> tuple[list[SourcedEffect], list[SourcedEffect]]:
    entity_manager.id_card_target = id_target

    return [], []


def _apply_target_card_clear(
    entity_manager: EntityManager,
) -> tuple[list[SourcedEffect], list[SourcedEffect]]:
    entity_manager.id_card_target = None

    return [], []


def _apply_deal_damage(
    entity_manager: EntityManager, id_source: int, id_target: int, value: int
) -> tuple[list[SourcedEffect], list[SourcedEffect]]:
    source = entity_manager.entities[id_source]

    # TODO: think if there's a better solution
    if isinstance(source, Card):
        source = entity_manager.entities[entity_manager.id_character]

    target = entity_manager.entities[id_target]

    # Apply strength
    if ModifierType.STRENGTH in source.modifiers:
        value += source.modifiers[ModifierType.STRENGTH].stacks_current

    # Apply weak
    if ModifierType.WEAK in source.modifiers:
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
) -> tuple[list[SourcedEffect], list[SourcedEffect]]:
    target = entity_manager.entities[id_target]

    target.block_current = min(target.block_current + value, BLOCK_MAX)
    return [], []


def _apply_play_card(
    entity_manager: EntityManager, id_target: int
) -> tuple[list[SourcedEffect], list[SourcedEffect]]:
    target = entity_manager.entities[id_target]

    return (
        [],
        [
            # TODO; move to engine?
            SourcedEffect(Effect(EffectType.DECREASE_ENERGY, value=target.cost)),
            SourcedEffect(Effect(EffectType.DISCARD), id_target=id_target),
            *[SourcedEffect(effect, id_source=id_target) for effect in target.effects],
        ],
    )


# TODO: handle infinite loop
def _apply_draw_card(
    entity_manager: EntityManager, amount: int
) -> tuple[list[SourcedEffect], list[SourcedEffect]]:
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
) -> tuple[list[SourcedEffect], list[SourcedEffect]]:
    energy = entity_manager.entities[entity_manager.id_energy]
    energy.current = energy.max

    return [], []


def _apply_decrease_energy(
    entity_manager: EntityManager, value: int
) -> tuple[list[SourcedEffect], list[SourcedEffect]]:
    energy = entity_manager.entities[entity_manager.id_energy]

    if energy.current < value:
        raise ValueError(f"Can't dercrease current energy ({energy.current}) by {value}")

    energy.current = energy.current - value

    return [], []


def _apply_gain_weak(
    entity_manager: EntityManager, id_target: int, value: int
) -> tuple[list[SourcedEffect], list[SourcedEffect]]:
    target = entity_manager.entities[id_target]
    if ModifierType.WEAK in target.modifiers:
        target.modifiers[ModifierType.WEAK].stacks_current += value

        return [], []

    target.modifiers[ModifierType.WEAK] = create_modifier_weak(value)

    return [], []


def _apply_gain_strength(
    entity_manager: EntityManager, id_target: int, value: int
) -> tuple[list[SourcedEffect], list[SourcedEffect]]:
    target = entity_manager.entities[id_target]
    if ModifierType.STRENGTH in target.modifiers:
        target.modifiers[ModifierType.STRENGTH].stacks_current += value

        return [], []

    target.modifiers[ModifierType.STRENGTH] = create_modifier_strength(value)

    return [], []


def _apply_mod_tick(
    entity_manager: EntityManager, id_target: int
) -> tuple[list[SourcedEffect], list[SourcedEffect]]:
    target = entity_manager.entities[id_target]

    for modifier_type, modifier in list(target.modifiers.items()):
        if modifier.stacks_duration:
            modifier.stacks_current -= 1

            if modifier.stacks_current < modifier.stacks_min:
                del target.modifiers[modifier_type]

    return [], []


def _apply_discard(
    entity_manager: EntityManager, id_target: int
) -> tuple[list[SourcedEffect], list[SourcedEffect]]:
    entity_manager.id_cards_in_hand.remove(id_target)
    entity_manager.id_cards_in_disc_pile.append(id_target)

    return [], []


def _apply_zero_block(
    entity_manager: EntityManager, id_target: int
) -> tuple[list[SourcedEffect], list[SourcedEffect]]:
    target = entity_manager.entities[id_target]

    target.block_current = 0

    return [], []


def _apply_shuffle_deck_into_draw_pile(
    entity_manager: EntityManager,
) -> tuple[list[SourcedEffect], list[SourcedEffect]]:
    entity_manager.id_cards_in_draw_pile = entity_manager.id_cards_in_deck.copy()
    random.shuffle(entity_manager.id_cards_in_draw_pile)

    return [], []


def _apply_update_move(
    entity_manager: EntityManager, id_target: int
) -> tuple[list[SourcedEffect], list[SourcedEffect]]:
    target = entity_manager.entities[id_target]

    move_new = ais[target.name](target.move_current, target.move_history)
    target.move_current = move_new
    target.move_history.append(move_new)

    return [], []
