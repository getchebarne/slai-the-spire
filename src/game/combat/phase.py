import random

from src.game.combat.ai import ais
from src.game.combat.effect_queue import EffectQueue
from src.game.combat.entities import Character
from src.game.combat.entities import Effect
from src.game.combat.entities import EffectTargetType
from src.game.combat.entities import EffectType
from src.game.combat.entities import Entities
from src.game.combat.entities import Monster


def combat_start(entities: Entities, effect_queue: EffectQueue) -> None:
    # Shuffle deck into draw pile
    entities.card_in_draw_pile_ids = list(entities.card_in_deck_ids)
    random.shuffle(entities.card_in_draw_pile_ids)

    # Get first move from monsters. TODO: revisit
    for monster in entities.get_monsters():
        ais[monster.name](monster)

    # Set start of turn to character & call it's turn start
    entities.actor_turn_id = entities.character_id
    _queue_turn_start_effects(entities, effect_queue, entities.character_id)

    # TODO: relic start of combat effects


def _queue_turn_start_effects(
    entities: Entities, effect_queue: EffectQueue, actor_id: int
) -> None:
    # Common effects
    effects = [Effect(EffectType.ZERO_BLOCK, target_type=EffectTargetType.SOURCE)]

    # Character and monster-specific effects
    actor = entities.get_entity(actor_id)
    if isinstance(actor, Character):
        # Draw 5 cards and refill energy
        effects += [Effect(EffectType.DRAW_CARD, 5), Effect(EffectType.REFILL_ENERGY)]

    elif isinstance(actor, Monster):
        # TODO: empty for now
        pass

    # Process effects
    effect_queue.add_to_bot(actor_id, *effects)


def _queue_turn_end_effects(entities: Entities, effect_queue: EffectQueue, actor_id: int) -> None:
    actor = entities.get_entity(actor_id)

    # Common. TODO: function
    for modifier_type, modifier in actor.modifiers.items():
        if modifier.stacks_duration:
            modifier.stacks -= 1

    actor.modifiers = {
        modifier_type: modifier
        for modifier_type, modifier in actor.modifiers.items()
        if modifier.stacks > modifier.stacks_min
    }
    effects = []
    if isinstance(actor, Character):
        # Character-specific effects
        effect_queue.add_to_bot(
            actor_id,
            Effect(EffectType.DISCARD, target_type=EffectTargetType.CARD_IN_HAND),
        )

    elif isinstance(actor, Monster):
        # TODO: no effects for now
        pass

    # Process effects
    effect_queue.add_to_bot(actor_id, *effects)
