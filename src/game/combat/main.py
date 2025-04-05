from typing import Callable

from src.agents.pseudorandom import select_action
from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.create import create_combat_state
from src.game.combat.drawer import draw_combat
from src.game.combat.effect import Effect
from src.game.combat.effect import EffectType
from src.game.combat.effect import SourcedEffect
from src.game.combat.effect_queue import add_to_bot
from src.game.combat.effect_queue import process_effect_queue
from src.game.combat.phase import get_end_of_turn_effects
from src.game.combat.phase import get_start_of_combat_effects
from src.game.combat.phase import get_start_of_turn_effects
from src.game.combat.state import CombatState
from src.game.combat.utils import card_requires_target
from src.game.combat.utils import is_game_over
from src.game.combat.view import CombatView
from src.game.combat.view import view_combat


class InvalidActionError(Exception):
    pass


def _handle_end_turn(cs: CombatState) -> list[SourcedEffect]:
    # Character's turn end
    sourced_effects = get_end_of_turn_effects(cs.entity_manager, cs.entity_manager.id_character)

    # Monsters
    for id_monster in cs.entity_manager.id_monsters:
        monster = cs.entity_manager.entities[id_monster]

        # Turn start
        sourced_effects += get_start_of_turn_effects(cs.entity_manager, id_monster)

        # Move's effects
        sourced_effects += [
            SourcedEffect(effect, id_monster) for effect in monster.move_current.effects
        ]

        # Update move
        sourced_effects += [SourcedEffect(Effect(EffectType.UPDATE_MOVE), id_target=id_monster)]

        # Turn end
        sourced_effects += get_end_of_turn_effects(cs.entity_manager, id_monster)

    # Character's turn start
    sourced_effects += get_start_of_turn_effects(cs.entity_manager, cs.entity_manager.id_character)

    return sourced_effects


def _handle_select_entity(cs: CombatState, id_target: int) -> list[SourcedEffect]:
    cs.entity_manager.id_effect_target = None
    cs.entity_manager.id_card_target = None

    if cs.entity_manager.id_card_active is None and not cs.effect_queue:
        # Selected card
        card = cs.entity_manager.entities[id_target]
        if card_requires_target(card):
            cs.entity_manager.id_card_active = id_target
            cs.entity_manager.id_selectables = cs.entity_manager.id_monsters

            return []

        # Play the card
        return [
            SourcedEffect(Effect(EffectType.PLAY_CARD), cs.entity_manager.id_character, id_target)
        ]

    if cs.entity_manager.id_card_active is not None and not cs.effect_queue:
        # Selected the active card's target, play the card
        id_card_active = cs.entity_manager.id_card_active

        cs.entity_manager.id_card_active = None
        cs.entity_manager.id_card_target = id_target

        return [
            SourcedEffect(
                Effect(EffectType.PLAY_CARD),
                cs.entity_manager.id_character,
                id_card_active,
            )
        ]

    if cs.entity_manager.id_card_active is None and cs.effect_queue:
        # Selected the active effect's target
        cs.entity_manager.id_effect_target = id_target

        return []


def handle_action(cs: CombatState, action: Action) -> list[SourcedEffect]:
    if action.type == ActionType.END_TURN:
        if cs.entity_manager.id_card_active is not None or cs.effect_queue:
            raise InvalidActionError

        return _handle_end_turn(cs)

    elif action.type == ActionType.SELECT_ENTITY:
        return _handle_select_entity(cs, action.target_id)


def step(cs: CombatState, action: Action) -> None:
    # Handle action
    sourced_effects = handle_action(cs, action)
    add_to_bot(cs.effect_queue, *sourced_effects)

    # Process round
    id_queries = process_effect_queue(cs.entity_manager, cs.effect_queue)
    if id_queries is not None:
        cs.entity_manager.id_selectables = id_queries

        return

    # Back to default state
    if cs.entity_manager.id_card_active is None:
        cs.entity_manager.id_selectables = []
        for id_card_in_hand in cs.entity_manager.id_cards_in_hand:
            card = cs.entity_manager.entities[id_card_in_hand]
            if card.cost <= cs.entity_manager.entities[cs.entity_manager.id_energy].current:
                cs.entity_manager.id_selectables.append(id_card_in_hand)


def start_combat(cs: CombatState) -> None:
    # Queue start of combat effects and process them
    sourced_effects = get_start_of_combat_effects(cs.entity_manager)
    add_to_bot(cs.effect_queue, *sourced_effects)
    process_effect_queue(cs.entity_manager, cs.effect_queue)

    # TODO: find better way to do this
    cs.entity_manager.id_selectables = cs.entity_manager.id_cards_in_hand


# TODO: fixx
def main(cs: CombatState, select_action_fn: Callable[[CombatView], Action]) -> None:
    start_combat(cs)

    while not is_game_over(combat_state.entity_manager):
        # Get combat view and draw it on the terminal
        combat_view = view_combat(combat_state)
        draw_combat(combat_view)

        # Get action from agent
        action = select_action_fn(combat_view)

        # Game step
        step(cs, action)

    # TODO: combat end


if __name__ == "__main__":
    # Instance combat manager
    combat_state = create_combat_state()

    # Execute
    main(combat_state, select_action)
