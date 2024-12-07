from typing import Optional

from src.agents.random import BaseAgent
from src.agents.random import RandomAgent
from src.game.combat.action import Action
from src.game.combat.create import create_combat_manager
from src.game.combat.drawer import draw_combat
from src.game.combat.effect_queue import process_queue
from src.game.combat.entities import Card
from src.game.combat.entities import Effect
from src.game.combat.entities import EffectTargetType
from src.game.combat.entities import EffectType
from src.game.combat.handle_input import handle_action
from src.game.combat.manager import CombatManager
from src.game.combat.phase import combat_start
from src.game.combat.state import State
from src.game.combat.state import on_enter
from src.game.combat.utils import is_game_over
from src.game.combat.view import view_combat


def _card_requires_target(card: Card) -> bool:
    for effect in card.effects:
        if effect.target_type == EffectTargetType.CARD_TARGET:
            return True

    return False


# TODO: move to state.py
def _set_state(
    state: State, combat_manager: CombatManager, entity_selectable_ids: Optional[list[int]] = None
) -> None:
    on_enter(state, combat_manager.entities, entity_selectable_ids)
    combat_manager.state = state


# TODO: find more suitable name
def process(combat_manager: CombatManager) -> None:
    # Check if there's an active card
    card_active_id = combat_manager.entities.card_active_id
    if card_active_id is not None and card_active_id in combat_manager.entities.card_in_hand_ids:
        # Check if the card requires target
        if _card_requires_target(combat_manager.entities.get_entity(card_active_id)):
            if combat_manager.entities.card_target_id is None:
                # Need to wait for card target
                _set_state(State.AWAIT_CARD_TARGET, combat_manager)

                return

        # Play card
        combat_manager.effect_queue.add_to_bot(
            combat_manager.entities.character_id,
            Effect(EffectType.PLAY_CARD, target_type=EffectTargetType.CARD_ACTIVE),
        )

    # Process queue TODO: should be standalone
    query_ids = process_queue(combat_manager.entities, combat_manager.effect_queue)
    if query_ids is None:
        _set_state(State.DEFAULT, combat_manager)

        return

    _set_state(State.AWAIT_EFFECT_TARGET, combat_manager, query_ids)


def step(combat_manager: CombatManager, action: Action) -> None:
    # Handle action
    handle_action(combat_manager, action)

    # Process round
    process(combat_manager)


def main(combat_manager: CombatManager, agent: BaseAgent) -> None:
    # Combat start TODO: change name
    combat_start(combat_manager)
    process(combat_manager)

    while not is_game_over(combat_manager.entities):
        # Get combat view and draw it on the terminal
        combat_view = view_combat(combat_manager)
        draw_combat(combat_view)

        # Get action from agent
        action = agent.select_action(combat_view)

        # Game step
        step(combat_manager, action)

    # TODO: combat end


if __name__ == "__main__":
    # Instance combat manager
    combat_manager = create_combat_manager()

    # Instance agent
    agent = RandomAgent()

    # Execute
    main(combat_manager, agent)
