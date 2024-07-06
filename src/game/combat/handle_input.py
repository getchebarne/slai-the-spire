from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.effect_queue import EffectQueue
from src.game.combat.effect_queue import process_queue
from src.game.combat.phase import turn_switch
from src.game.combat.state import Card
from src.game.combat.state import Effect
from src.game.combat.state import EffectTargetType
from src.game.combat.state import EffectType
from src.game.combat.state import GameState
from src.game.combat.state import Monster
from src.game.combat.utils import card_requires_target


# TODO: improve this function
def handle_action(state: GameState, effect_queue: EffectQueue, action: Action) -> None:
    if action.type == ActionType.END_TURN:
        state.card_active_id = None
        turn_switch(state, effect_queue)
        return

    if action.type == ActionType.SELECT_ENTITY:
        if effect_queue.effect_pending is not None:
            state.selected_entity_ids = [action.target_id]
            process_queue(state, effect_queue)
            return

        # Get target entity. TODO: change to `in` check
        target = state.get_entity(action.target_id)

        # If it's a card, set it as active
        if isinstance(target, Card):
            state.card_active_id = action.target_id
            if card_requires_target(state.get_active_card()):
                # Wait for player input
                return

        # If it's a monster, set it as target
        if isinstance(target, Monster):
            state.card_target_id = action.target_id

    # Play card. TODO: confirm?
    effect_queue.add_to_bot(
        None, Effect(EffectType.PLAY_CARD, target_type=EffectTargetType.CARD_ACTIVE)
    )
    process_queue(state, effect_queue)

    # Untag card
    state.card_active_id = None
