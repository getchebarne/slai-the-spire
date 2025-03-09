from dataclasses import replace

from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.ai import ais
from src.game.combat.phase import queue_turn_end_effects
from src.game.combat.phase import queue_turn_start_effects
from src.game.combat.state import CombatState
from src.game.combat.state import add_to_bot


class InvalidActionError(Exception):
    pass


def _handle_end_turn(combat_state: CombatState) -> CombatState:
    if combat_state.entities.card_active_id is not None or combat_state.effect_queue:
        raise InvalidActionError

    # Character's turn end
    entities_new, effect_queue_new = queue_turn_end_effects(
        combat_state.entities, combat_state.effect_queue, combat_state.entities.character_id
    )
    combat_state.entities = entities_new
    combat_state.effect_queue = effect_queue_new

    for monster_id in combat_state.entities.monster_ids:
        monster = combat_state.entities.all[monster_id]

        # Monster's turn start
        combat_state.effect_queue = queue_turn_start_effects(
            combat_state.entities, combat_state.effect_queue, monster_id
        )

        # Queue monster's move's effects
        combat_state.effect_queue = add_to_bot(
            combat_state.effect_queue, monster_id, *monster.move_current.effects
        )

        # Update monster's move
        monster = replace(
            monster, move_current=ais[monster.name](monster.move_current, monster.move_history)
        )
        # TODO: improve this
        monster = replace(monster, move_history=monster.move_history + [monster.move_current])

        # Monster's turn end
        entities_new, effect_queue_new = queue_turn_end_effects(
            combat_state.entities, combat_state.effect_queue, monster_id
        )
        combat_state.entities = entities_new
        combat_state.effect_queue = effect_queue_new

    # Character's turn start
    combat_state.effect_queue = queue_turn_start_effects(
        combat_state.entities, combat_state.effect_queue, combat_state.entities.character_id
    )

    return combat_state


def _handle_select_entity(combat_state: CombatState, target_id: int) -> CombatState:
    if combat_state.entities.card_active_id is None and not combat_state.effect_queue:
        # Set active card in `combat_state.entities`
        combat_state.entities = replace(combat_state.entities, card_active_id=target_id)

        return combat_state

    if combat_state.entities.card_active_id is not None and not combat_state.effect_queue:
        # Set card target in `combat_state.entities`
        combat_state.entities = replace(combat_state.entities, card_target_id=target_id)

        return combat_state

    if combat_state.entities.card_active_id is None and combat_state.effect_queue:
        # Set effect target in `combat_state.entities`
        combat_state.entities = replace(combat_state.entities, effect_target_id=target_id)

        return combat_state


def handle_action(combat_state: CombatState, action: Action) -> None:
    if action.type == ActionType.END_TURN:
        _handle_end_turn(combat_state)

    elif action.type == ActionType.SELECT_ENTITY:
        _handle_select_entity(combat_state, action.target_id)
