from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.ai import ais
from src.game.combat.effect_queue import process_queue
from src.game.combat.entities import Card
from src.game.combat.entities import Effect
from src.game.combat.entities import EffectTargetType
from src.game.combat.entities import EffectType
from src.game.combat.manager import CombatManager
from src.game.combat.phase import _queue_turn_end_effects
from src.game.combat.phase import _queue_turn_start_effects
from src.game.combat.state import State
from src.game.combat.state import on_enter


class InvalidActionError(Exception):
    pass


def _card_requires_target(card: Card) -> bool:
    for effect in card.effects:
        if effect.target_type == EffectTargetType.CARD_TARGET:
            return True

    return False


def _handle_end_turn(combat_manager: CombatManager) -> State:
    if combat_manager.state == State.AWAIT_EFFECT_TARGET:
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
        combat_manager.effect_queue.add_to_bot(monster_id, *monster.move.effects)

        # Update monster's move
        ais[monster.name](monster)

        # Monster's turn end
        _queue_turn_end_effects(combat_manager.entities, combat_manager.effect_queue, monster_id)

    # Character's turn start
    _queue_turn_start_effects(
        combat_manager.entities, combat_manager.effect_queue, combat_manager.entities.character_id
    )

    # Process queue
    query_ids = process_queue(combat_manager.entities, combat_manager.effect_queue)
    if query_ids is not None:
        combat_manager.entities.entity_selectable_ids = query_ids

        return State.AWAIT_EFFECT_TARGET

    return State.DEFAULT


def _handle_select_entity(combat_manager: CombatManager, target_id: int) -> State:
    if combat_manager.state == State.DEFAULT:
        # Set active card in `combat_manager.entities`
        combat_manager.entities.card_active_id = target_id

        if _card_requires_target(combat_manager.entities.get_entity(target_id)):
            return State.AWAIT_CARD_TARGET

        # Play card
        combat_manager.effect_queue.add_to_bot(
            combat_manager.entities.character_id,
            Effect(EffectType.PLAY_CARD, target_type=EffectTargetType.CARD_ACTIVE),
        )

    elif combat_manager.state == State.AWAIT_CARD_TARGET:
        # Set card target in `combat_manager.entities`
        combat_manager.entities.card_target_id = target_id

        # Play card
        combat_manager.effect_queue.add_to_bot(
            combat_manager.entities.character_id,
            Effect(EffectType.PLAY_CARD, target_type=EffectTargetType.CARD_ACTIVE),
        )

    elif combat_manager.state == State.AWAIT_EFFECT_TARGET:
        # Set effect target in `combat_manager.entities`
        combat_manager.entities.effect_target_id = target_id

    # Process queue
    query_ids = process_queue(combat_manager.entities, combat_manager.effect_queue)
    if query_ids is not None:
        combat_manager.entities.entity_selectable_ids = query_ids

        return State.AWAIT_EFFECT_TARGET

    return State.DEFAULT


# TODO: improve this function
def handle_action(combat_manager: CombatManager, action: Action) -> None:
    if action.type == ActionType.END_TURN:
        new_state = _handle_end_turn(combat_manager)

    elif action.type == ActionType.SELECT_ENTITY:
        new_state = _handle_select_entity(combat_manager, action.target_id)

    # Enter new state
    on_enter(new_state, combat_manager.entities)
    combat_manager.state = new_state
