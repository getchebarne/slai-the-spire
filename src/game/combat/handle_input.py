from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.effect_queue import EffectQueue
from src.game.combat.effect_queue import process_queue
from src.game.combat.entities import Card
from src.game.combat.entities import Effect
from src.game.combat.entities import EffectTargetType
from src.game.combat.entities import EffectType
from src.game.combat.entities import Entities
from src.game.combat.entities import Monster
from src.game.combat.phase import turn_switch
from src.game.combat.utils import card_requires_target


# TODO: improve this function
def handle_action(entities: Entities, effect_queue: EffectQueue, action: Action) -> None:
    if action.type == ActionType.END_TURN:
        entities.card_active_id = None
        turn_switch(entities, effect_queue)
        return

    if action.type == ActionType.SELECT_ENTITY:
        if effect_queue.get_pending() is not None:
            entities.entity_selected_ids = [action.target_id]
            process_queue(entities, effect_queue)
            entities.entity_selected_ids = None
            return

        # Get target entity. TODO: change to `in` check
        target = entities.get_entity(action.target_id)

        # If it's a card, set it as active
        if isinstance(target, Card):
            entities.card_active_id = action.target_id
            if card_requires_target(entities.get_active_card()):
                # Wait for player input
                return

        # If it's a monster, set it as target
        if isinstance(target, Monster):
            entities.card_target_id = action.target_id

    # Play card. TODO: confirm?
    effect_queue.add_to_bot(
        None, Effect(EffectType.PLAY_CARD, target_type=EffectTargetType.CARD_ACTIVE)
    )
    process_queue(entities, effect_queue)

    # Untag card
    entities.card_active_id = None
