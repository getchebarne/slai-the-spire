import random
from dataclasses import replace

from src.game.combat.ai import ais
from src.game.combat.entities import Character
from src.game.combat.entities import Effect
from src.game.combat.entities import EffectTargetType
from src.game.combat.entities import EffectType
from src.game.combat.entities import Entities
from src.game.combat.entities import Monster
from src.game.combat.state import CombatState
from src.game.combat.state import QueuedEffect
from src.game.combat.state import add_to_bot


def start_combat(combat_state: CombatState) -> CombatState:
    # Shuffle deck into draw pile
    card_in_draw_pile_ids = list(combat_state.entities.card_in_deck_ids).copy()
    random.shuffle(card_in_draw_pile_ids)
    combat_state.entities = replace(
        combat_state.entities, card_in_draw_pile_ids=card_in_draw_pile_ids
    )

    # Get first move from monsters. TODO: revisit
    for monster_id in combat_state.entities.monster_ids:
        monster = combat_state.entities.all[monster_id]
        monster = replace(
            monster, move_current=ais[monster.name](monster.move_current, monster.move_history)
        )
        # TODO: improve
        monster = replace(monster, move_history=monster.move_history + [monster.move_current])

        all_new = combat_state.entities.all.copy()
        all_new[monster_id] = monster
        combat_state.entities = replace(combat_state.entities, all=all_new)

    # Set start of turn to character & call its turn start
    # TODO: do I need actor_turn_id?
    combat_state.entities = replace(
        combat_state.entities, actor_turn_id=combat_state.entities.character_id
    )
    combat_state.effect_queue = queue_turn_start_effects(
        combat_state.entities, combat_state.effect_queue, combat_state.entities.character_id
    )

    return combat_state


def queue_turn_start_effects(
    entities: Entities, effect_queue: list[QueuedEffect], actor_id: int
) -> list[QueuedEffect]:
    # Common effects
    effects = [Effect(EffectType.ZERO_BLOCK, target_type=EffectTargetType.SOURCE)]

    # Character and monster-specific effects
    actor = entities.all[actor_id]
    if isinstance(actor, Character):
        # Draw 5 cards and refill energy
        effects += [Effect(EffectType.DRAW_CARD, 5), Effect(EffectType.REFILL_ENERGY)]

    elif isinstance(actor, Monster):
        # TODO: empty for now
        pass

    # Queue effects
    return add_to_bot(effect_queue, actor_id, *effects)


def queue_turn_end_effects(
    entities: Entities, effect_queue: list[QueuedEffect], actor_id: int
) -> None:
    actor = entities.all[actor_id]

    # Common TODO: reenable
    # effects = [Effect(EffectType.MOD_TICK, target_type=EffectTargetType.SOURCE)]

    if isinstance(actor, Character):
        # Character-specific effects
        effect_queue_new = add_to_bot(
            effect_queue,
            actor_id,
            Effect(EffectType.DISCARD, target_type=EffectTargetType.CARD_IN_HAND),
        )

    elif isinstance(actor, Monster):
        # TODO: no effects for now
        effect_queue_new = effect_queue.copy()
        pass

    # Queue effects
    # effect_queue_new = add_to_bot(effect_queue_new, actor_id, *effects)

    return entities, effect_queue_new
