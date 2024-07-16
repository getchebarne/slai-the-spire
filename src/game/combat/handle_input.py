from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.ai import ais
from src.game.combat.effect_queue import EffectQueue
from src.game.combat.effect_queue import process_queue
from src.game.combat.entities import Card
from src.game.combat.entities import Effect
from src.game.combat.entities import EffectTargetType
from src.game.combat.entities import EffectType
from src.game.combat.entities import Entities
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


def _handle_end_turn(entities: Entities, effect_queue: EffectQueue, state: State) -> State:
    if state == State.AWAIT_EFFECT_TARGET:
        raise InvalidActionError

    # Character's turn end
    _queue_turn_end_effects(entities, effect_queue, entities.character_id)

    for monster_id in entities.monster_ids:
        monster = entities.get_entity(monster_id)

        # Monster's turn start
        _queue_turn_start_effects(entities, effect_queue, monster_id)

        # Queue monster's move's effects
        effect_queue.add_to_bot(monster_id, *monster.move.effects)

        # Update monster's move
        ais[monster.name](monster)

        # Monster's turn end
        _queue_turn_end_effects(entities, effect_queue, monster_id)

    # Character's turn start
    _queue_turn_start_effects(entities, effect_queue, entities.character_id)

    # Process queue
    query_ids = process_queue(entities, effect_queue)
    if query_ids is not None:
        entities.entity_selectable_ids = query_ids

        return State.AWAIT_EFFECT_TARGET

    return State.DEFAULT


def _handle_select_entity(
    entities: Entities, effect_queue: EffectQueue, state: State, target_id: int
) -> State:
    if state == State.DEFAULT:
        # Set active card in `entities`
        entities.card_active_id = target_id

        if _card_requires_target(entities.get_entity(target_id)):
            return State.AWAIT_CARD_TARGET

        # Play card
        effect_queue.add_to_bot(
            entities.character_id,
            Effect(EffectType.PLAY_CARD, target_type=EffectTargetType.CARD_ACTIVE),
        )

    elif state == State.AWAIT_CARD_TARGET:
        # Set card target in `entities`
        entities.card_target_id = target_id

        # Play card
        effect_queue.add_to_bot(
            entities.character_id,
            Effect(EffectType.PLAY_CARD, target_type=EffectTargetType.CARD_ACTIVE),
        )

    elif state == State.AWAIT_EFFECT_TARGET:
        # Set effect target in `entities`
        entities.effect_target_id = target_id

    # Process queue
    query_ids = process_queue(entities, effect_queue)
    if query_ids is not None:
        entities.entity_selectable_ids = query_ids

        return State.AWAIT_EFFECT_TARGET

    return State.DEFAULT


# TODO: improve this function
def handle_action(
    entities: Entities, effect_queue: EffectQueue, state: State, action: Action
) -> State:
    if action.type == ActionType.END_TURN:
        new_state = _handle_end_turn(entities, effect_queue, state)

    elif action.type == ActionType.SELECT_ENTITY:
        new_state = _handle_select_entity(entities, effect_queue, state, action.target_id)

    # Enter new state
    on_enter(new_state, entities)

    return new_state
