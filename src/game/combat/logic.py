import random

from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.context import Effect
from src.game.combat.context import EffectSelectionType
from src.game.combat.context import EffectTargetType
from src.game.combat.context import EffectType
from src.game.combat.context import Entity
from src.game.combat.context import GameContext
from src.game.combat.processors import get_effect_processors
from src.game.combat.utils import add_effects_to_bot
from src.game.combat.utils import card_requires_target
from src.game.combat.utils import get_active_card


# TODO: should be EffectPlayCard
def play_card(context: GameContext) -> None:
    active_card = get_active_card(context)
    if active_card.cost > context.energy.current:
        raise ValueError(f"Can't play card {active_card} with {context.energy.current} energy")

    # Subtract energy
    context.energy.current -= active_card.cost

    # Send card to discard pile
    context.hand.remove(active_card)
    context.discard_pile.add(active_card)


def _resolve_effect_target_type(
    effect_target_type: EffectTargetType, context: GameContext
) -> list[Entity]:
    if effect_target_type == EffectTargetType.CHARACTER:
        return [context.character]

    if effect_target_type == EffectTargetType.MONSTER:
        return context.monsters

    if effect_target_type == EffectTargetType.CARD_TARGET:
        return [context.card_target]

    if effect_target_type == EffectTargetType.CARD_IN_HAND:
        return context.hand

    raise ValueError(f"Unsupported effect target type: {effect_target_type}")


def _resolve_effect_selection_type(
    effect_selection_type: EffectSelectionType, entities: list[Entity]
) -> list[Entity]:
    if effect_selection_type == effect_selection_type.SPECIFIC:
        # TODO: get action from agent
        pass

    if effect_selection_type == EffectSelectionType.ALL:
        return entities

    if effect_selection_type == EffectSelectionType.RANDOM:
        return [random.choice(entities)]

    raise ValueError(f"Unsupported effect selection type: {effect_selection_type}")


def get_effect_targets(
    effect_target_type: EffectTargetType,
    effect_selection_type: EffectSelectionType,
    context: GameContext,
) -> list[Entity]:
    if effect_target_type is None:
        return []

    query_entity_ids = _resolve_effect_target_type(effect_target_type, context)

    if effect_selection_type is None:
        return query_entity_ids

    return _resolve_effect_selection_type(effect_selection_type, query_entity_ids)


def process_next_effect(context: GameContext) -> None:
    # Get effect from queue
    effect = context.queue.popleft()

    # Get effect's targets and processors
    targets = get_effect_targets(effect.target_type, effect.selection_type, context)
    processors = get_effect_processors(effect.type)

    # Execute
    if not targets:
        targets = [1]  # TODO: fix

    for target in targets:
        # Set target and value
        context.effect_target = target
        context.effect_value = effect.value

        # Run effect's processors
        for processor in processors:
            processor(context)


def _handle_action_select_card(context: GameContext, card_idx: int) -> None:
    context.active_card = context.hand[card_idx]


def _handle_action_select_monster(context: GameContext, monster_idx: int) -> None:
    context.card_target = context.monsters[monster_idx]


def is_game_over(context: GameContext) -> bool:
    return context.character.health.current <= 0 or all(
        [monster.health.current <= 0 for monster in context.monsters]
    )


def character_turn_start(context: GameContext) -> None:
    add_effects_to_bot(context, Effect(EffectType.DRAW_CARD, 5), Effect(EffectType.REFILL_ENERGY))
    while context.queue:
        process_next_effect(context)


def character_turn(context: GameContext, action: Action) -> None:
    if action.type == ActionType.SELECT_CARD:
        card = context.hand[action.index]
        card.is_active = True
        if card_requires_target(card):
            # Wait for player input
            return

    if action.type == ActionType.SELECT_MONSTER:
        context.card_target = context.monsters[action.index]

    # Play card. TODO: confirm?
    card = get_active_card(context)
    play_card(context)
    add_effects_to_bot(context, *card.effects)

    # Untag card
    card.is_active = False
    while context.queue:
        process_next_effect(context)


def combat_start(context: GameContext) -> None:
    context.draw_pile = list(context.deck)
    random.shuffle(context.draw_pile)


def character_turn_end(context: GameContext) -> None:
    add_effects_to_bot(
        context,
        Effect(
            EffectType.DISCARD,
            target_type=EffectTargetType.CARD_IN_HAND,
            selection_type=EffectSelectionType.ALL,
        ),
    )
    while context.queue:
        process_next_effect(context)
