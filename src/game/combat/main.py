from dataclasses import replace

from src.agents.random import BaseAgent
from src.agents.random import RandomAgent
from src.game.combat.action import Action
from src.game.combat.create import create_combat_state
from src.game.combat.drawer import draw_combat
from src.game.combat.effect_queue import process_queue
from src.game.combat.entities import Card
from src.game.combat.entities import Effect
from src.game.combat.entities import EffectTargetType
from src.game.combat.entities import EffectType
from src.game.combat.handle_input import handle_action
from src.game.combat.phase import start_combat
from src.game.combat.state import CombatState
from src.game.combat.state import add_to_bot
from src.game.combat.utils import is_game_over
from src.game.combat.view import view_combat


def _card_requires_target(card: Card) -> bool:
    for effect in card.effects:
        if effect.target_type == EffectTargetType.CARD_TARGET:
            return True

    return False


# TODO: find more suitable name
def process(combat_state: CombatState) -> CombatState:
    # Check if there's an active card
    card_active_id = combat_state.entities.card_active_id
    # TODO: a card shouldn't be able to be active and not in the hand at the same time
    if card_active_id is not None and card_active_id in combat_state.entities.card_in_hand_ids:
        # Check if the card requires target
        if _card_requires_target(combat_state.entities.all[card_active_id]):
            if combat_state.entities.card_target_id is None:
                # Need to wait for card target
                combat_state.entities = replace(
                    combat_state.entities, entity_selectable_ids=combat_state.entities.monster_ids
                )

                return combat_state

        # Play card
        combat_state.effect_queue = add_to_bot(
            combat_state.effect_queue,
            combat_state.entities.character_id,
            Effect(EffectType.PLAY_CARD, target_type=EffectTargetType.CARD_ACTIVE),
        )

    # Process queue
    entities_new, effect_queue_new = process_queue(
        combat_state.entities, combat_state.effect_queue
    )
    if effect_queue_new:
        # TODO: this shouldn't happen here
        entities_new = replace(entities_new, card_active_id=None)

        return CombatState(entities_new, effect_queue_new)

    # Reset active card, card target, and effect target
    entities_new = replace(
        entities_new,
        card_active_id=None,
        card_target_id=None,
        effect_target_id=None,
        entity_selectable_ids=[
            card_in_hand_id
            for card_in_hand_id in entities_new.card_in_hand_ids
            if entities_new.all[card_in_hand_id].cost
            <= entities_new.all[entities_new.energy_id].current
        ],
    )
    return CombatState(entities_new, effect_queue_new)


def step(combat_state: CombatState, action: Action) -> None:
    # Handle action
    handle_action(combat_state, action)

    # Process round
    combat_state_new = process(combat_state)

    return combat_state_new


def main(combat_state: CombatState, agent: BaseAgent) -> None:
    # Combat start TODO: change name
    combat_state = start_combat(combat_state)
    combat_state = process(combat_state)

    while not is_game_over(combat_state.entities):
        # Get combat view and draw it on the terminal
        combat_view = view_combat(combat_state)
        draw_combat(combat_view)

        # Get action from agent
        action = agent.select_action(combat_view)

        # Game step
        combat_state = step(combat_state, action)

    # TODO: combat end


if __name__ == "__main__":
    # Instance combat manager
    combat_state = create_combat_state()

    # Instance agent
    agent = RandomAgent()

    # Execute
    main(combat_state, agent)
