import random
from dataclasses import replace

from src.game.ai.registry import AI_REGISTRY
from src.game.combat.effect import Effect
from src.game.combat.effect import EffectTargetType
from src.game.combat.effect import EffectType
from src.game.combat.phase import get_end_of_turn_effects
from src.game.combat.phase import get_start_of_turn_effects
from src.game.entity.actor import ModifierType
from src.game.entity.card import EntityCard
from src.game.entity.manager import EntityManager
from src.game.entity.monster import EntityMonster
from src.game.factory.modifier.ritual import create_modifier_data_ritual
from src.game.factory.modifier.strength import create_modifier_data_strength
from src.game.factory.modifier.vulnerable import create_modifier_data_vulnerable
from src.game.factory.modifier.weak import create_modifier_data_weak


WEAK_FACTOR = 0.75
VULN_FACTOR = 1.50
BLOCK_MAX = 999


def apply_effect(
    entity_manager: EntityManager,
    effect_type: EffectType,
    effect_value: int | None,
    id_source: int | None,
    id_target: int | None,
) -> tuple[list[Effect], list[Effect]]:
    if effect_type == EffectType.DEAL_DAMAGE:
        return _apply_deal_damage(entity_manager, id_source, id_target, effect_value)

    if effect_type == EffectType.LOSE_HP:
        return _apply_lose_hp(entity_manager, id_target, effect_value)

    if effect_type == EffectType.GAIN_VULNERABLE:
        return _apply_gain_vulnerable(entity_manager, id_target, effect_value)

    if effect_type == EffectType.GAIN_RITUAL:
        return _apply_gain_ritual(entity_manager, id_target, effect_value)

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
) -> tuple[list[Effect], list[Effect]]:
    # Character's turn end
    effects = get_end_of_turn_effects(entity_manager, entity_manager.id_character)

    # Monsters
    for id_monster in entity_manager.id_monsters:
        monster = entity_manager.entities[id_monster]

        # Turn start
        effects += get_start_of_turn_effects(entity_manager, id_monster)

        # Move's effects
        effects += [
            replace(effect, id_source=id_monster)
            for effect in monster.move_map[monster.move_name_current]
        ]

        # Update move
        effects += [Effect(EffectType.UPDATE_MOVE, id_target=id_monster)]

        # Turn end
        effects += get_end_of_turn_effects(entity_manager, id_monster)

    # Character's turn start
    effects += get_start_of_turn_effects(entity_manager, entity_manager.id_character)

    return effects, []


def _apply_card_active_set(
    entity_manager: EntityManager, id_target: int
) -> tuple[list[Effect], list[Effect]]:
    entity_manager.id_card_active = id_target

    return [], []


def _apply_card_active_clear(
    entity_manager: EntityManager,
) -> tuple[list[Effect], list[Effect]]:
    entity_manager.id_card_active = None

    return [], []


def _apply_target_effect_set(
    entity_manager: EntityManager, id_target: int
) -> tuple[list[Effect], list[Effect]]:
    entity_manager.id_effect_target = id_target

    return [], []


def _apply_target_effect_clear(
    entity_manager: EntityManager,
) -> tuple[list[Effect], list[Effect]]:
    entity_manager.id_effect_target = None

    return [], []


def _apply_target_card_set(
    entity_manager: EntityManager, id_target: int
) -> tuple[list[Effect], list[Effect]]:
    entity_manager.id_card_target = id_target

    return [], []


def _apply_target_card_clear(
    entity_manager: EntityManager,
) -> tuple[list[Effect], list[Effect]]:
    entity_manager.id_card_target = None

    return [], []


def _apply_deal_damage(
    entity_manager: EntityManager, id_source: int, id_target: int, value: int
) -> tuple[list[Effect], list[Effect]]:
    source = entity_manager.entities[id_source]

    # TODO: think if there's a better solution
    if isinstance(source, EntityCard):
        source = entity_manager.entities[entity_manager.id_character]

    target = entity_manager.entities[id_target]

    # Apply strength
    if ModifierType.STRENGTH in source.modifier_map:
        value += source.modifier_map[ModifierType.STRENGTH].stacks_current

    # Apply weak
    if ModifierType.WEAK in source.modifier_map:
        value *= WEAK_FACTOR

    # Apply vulnerable
    if ModifierType.VULNERABLE in target.modifier_map:
        value *= VULN_FACTOR

    # Calculate damage over block
    value = int(value)
    damage_over_block = max(0, value - target.block_current)

    # Remove block
    target.block_current = max(0, target.block_current - value)

    # Return a top effect to subtract the damage over block from the target's current health
    return [], [Effect(EffectType.LOSE_HP, value=damage_over_block, id_target=id_target)]


def _apply_lose_hp(
    entity_manager: EntityManager, id_target: int, value: int
) -> tuple[list[Effect], list[Effect]]:
    target = entity_manager.entities[id_target]

    if value >= target.health_current:
        # Death
        if isinstance(target, EntityMonster):
            # TODO: delete instance from `entity_manager.entities`
            entity_manager.id_monsters.remove(id_target)

            effects_top = []
            for modifier_type, modifier_data in target.modifier_map.items():
                if modifier_type == ModifierType.SPORE_CLOUD:
                    effects_top.append(
                        Effect(
                            Effect(
                                EffectType.GAIN_VULNERABLE,
                                modifier_data.stacks_current,
                                EffectTargetType.CHARACTER,
                            )
                        )
                    )

            return [], effects_top

    target.health_current = max(0, target.health_current - value)

    return [], []


def _apply_gain_block(
    entity_manager: EntityManager, id_target: int, value: int
) -> tuple[list[Effect], list[Effect]]:
    target = entity_manager.entities[id_target]

    target.block_current = min(target.block_current + value, BLOCK_MAX)
    return [], []


def _apply_play_card(
    entity_manager: EntityManager, id_target: int
) -> tuple[list[Effect], list[Effect]]:
    target = entity_manager.entities[id_target]

    return (
        [],
        [
            # TODO; move to engine?
            Effect(EffectType.DECREASE_ENERGY, value=target.cost),
            Effect(EffectType.DISCARD, id_target=id_target),
            *[replace(effect, id_source=id_target) for effect in target.effects],
        ],
    )


# TODO: handle infinite loop
def _apply_draw_card(
    entity_manager: EntityManager, amount: int
) -> tuple[list[Effect], list[Effect]]:
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
) -> tuple[list[Effect], list[Effect]]:
    energy = entity_manager.entities[entity_manager.id_energy]
    energy.current = energy.max

    return [], []


def _apply_decrease_energy(
    entity_manager: EntityManager, value: int
) -> tuple[list[Effect], list[Effect]]:
    energy = entity_manager.entities[entity_manager.id_energy]

    if energy.current < value:
        raise ValueError(f"Can't dercrease current energy ({energy.current}) by {value}")

    energy.current = energy.current - value

    return [], []


def _apply_gain_weak(
    entity_manager: EntityManager, id_target: int, value: int
) -> tuple[list[Effect], list[Effect]]:
    target = entity_manager.entities[id_target]
    if ModifierType.WEAK in target.modifier_map:
        target.modifier_map[ModifierType.WEAK].stacks_current += value

        return [], []

    target.modifier_map[ModifierType.WEAK] = create_modifier_data_weak(value)

    return [], []


def _apply_gain_vulnerable(
    entity_manager: EntityManager, id_target: int, value: int
) -> tuple[list[Effect], list[Effect]]:
    target = entity_manager.entities[id_target]
    if ModifierType.VULNERABLE in target.modifier_map:
        target.modifier_map[ModifierType.VULNERABLE].stacks_current += value

        return [], []

    target.modifier_map[ModifierType.VULNERABLE] = create_modifier_data_vulnerable(value)

    return [], []


def _apply_gain_ritual(
    entity_manager: EntityManager, id_target: int, value: int
) -> tuple[list[Effect], list[Effect]]:
    target = entity_manager.entities[id_target]
    if ModifierType.RITUAL in target.modifier_map:
        target.modifier_map[ModifierType.RITUAL].stacks_current += value

        return [], []

    target.modifier_map[ModifierType.RITUAL] = create_modifier_data_ritual(value)

    return [], []


def _apply_gain_strength(
    entity_manager: EntityManager, id_target: int, value: int
) -> tuple[list[Effect], list[Effect]]:
    target = entity_manager.entities[id_target]
    if ModifierType.STRENGTH in target.modifier_map:
        target.modifier_map[ModifierType.STRENGTH].stacks_current += value

        return [], []

    target.modifier_map[ModifierType.STRENGTH] = create_modifier_data_strength(value)

    return [], []


def _apply_mod_tick(
    entity_manager: EntityManager, id_target: int
) -> tuple[list[Effect], list[Effect]]:
    target = entity_manager.entities[id_target]

    for modifier_type, modifier_data in list(target.modifier_map.items()):
        if modifier_data.stacks_duration:
            modifier_data.stacks_current -= 1

            if modifier_data.stacks_current < modifier_data.stacks_min:
                del target.modifier_map[modifier_type]

    return [], []


def _apply_discard(
    entity_manager: EntityManager, id_target: int
) -> tuple[list[Effect], list[Effect]]:
    entity_manager.id_cards_in_hand.remove(id_target)
    entity_manager.id_cards_in_disc_pile.append(id_target)

    return [], []


def _apply_zero_block(
    entity_manager: EntityManager, id_target: int
) -> tuple[list[Effect], list[Effect]]:
    target = entity_manager.entities[id_target]

    target.block_current = 0

    return [], []


def _apply_shuffle_deck_into_draw_pile(
    entity_manager: EntityManager,
) -> tuple[list[Effect], list[Effect]]:
    entity_manager.id_cards_in_draw_pile = entity_manager.id_cards_in_deck.copy()
    random.shuffle(entity_manager.id_cards_in_draw_pile)

    return [], []


def _apply_update_move(
    entity_manager: EntityManager, id_target: int
) -> tuple[list[Effect], list[Effect]]:
    target = entity_manager.entities[id_target]

    move_name_new = AI_REGISTRY[target.name](target)
    target.move_name_current = move_name_new
    target.move_name_history.append(move_name_new)

    return [], []
