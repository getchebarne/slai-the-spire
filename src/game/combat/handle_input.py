from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.ai import ais
from src.game.combat.manager import CombatManager
from src.game.combat.phase import _queue_turn_end_effects
from src.game.combat.phase import _queue_turn_start_effects
from src.game.combat.state import State


class InvalidActionError(Exception):
    pass


def _handle_end_turn(combat_manager: CombatManager) -> None:
    if combat_manager.state != State.DEFAULT:
        raise InvalidActionError

    # Character's turn end
    _queue_turn_end_effects(
        combat_manager.entities, combat_manager.effect_queue, combat_manager.entities.character_id
    )

    for monster_id in combat_manager.entities.monster_ids:
        monster = combat_manager.entities.get_entity(monster_id)

        # Monster's turn start
        _queue_turn_start_effects(combat_manager.entities, combat_manager.effect_queue, monster_id)

        # Queue monster's move's effects
        combat_manager.effect_queue.add_to_bot(
            monster_id, *monster.moves[monster.move_name_current]
        )

        # Update monster's move
        monster.move_name_current = ais[monster.name](
            monster.move_name_current, monster.move_name_history
        )
        monster.move_name_history.append(monster.move_name_current)  # TODO: improve

        # Monster's turn end
        _queue_turn_end_effects(combat_manager.entities, combat_manager.effect_queue, monster_id)

    # Character's turn start
    _queue_turn_start_effects(
        combat_manager.entities, combat_manager.effect_queue, combat_manager.entities.character_id
    )


def _handle_select_entity(combat_manager: CombatManager, target_id: int) -> None:
    if combat_manager.state == State.DEFAULT:
        # Set active card in `combat_manager.entities`
        combat_manager.entities.card_active_id = target_id

        return

    if combat_manager.state == State.AWAIT_CARD_TARGET:
        # Set card target in `combat_manager.entities`
        combat_manager.entities.card_target_id = target_id

        return

    if combat_manager.state == State.AWAIT_EFFECT_TARGET:
        # Set effect target in `combat_manager.entities`
        combat_manager.entities.effect_target_id = target_id

        return


def handle_action(combat_manager: CombatManager, action: Action) -> None:
    if action.type == ActionType.END_TURN:
        _handle_end_turn(combat_manager)

    elif action.type == ActionType.SELECT_ENTITY:
        _handle_select_entity(combat_manager, action.target_id)
