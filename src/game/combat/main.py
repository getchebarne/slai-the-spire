from src.agents.random import BaseAgent
from src.agents.random import RandomAgent
from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.create import create_combat_state
from src.game.combat.drawer import draw_combat
from src.game.combat.effect_queue import add_to_bot
from src.game.combat.effect_queue import process_effect_queue
from src.game.combat.entities import Card
from src.game.combat.entities import Effect
from src.game.combat.entities import EffectTargetType
from src.game.combat.entities import EffectType
from src.game.combat.phase import ToBeQueuedEffect  # TODO: define elsewhere
from src.game.combat.phase import get_end_of_turn_effects
from src.game.combat.phase import get_start_of_combat_effects
from src.game.combat.phase import get_start_of_turn_effects
from src.game.combat.state import CombatState
from src.game.combat.utils import is_game_over
from src.game.combat.view import view_combat


def _card_requires_target(card: Card) -> bool:
    for effect in card.effects:
        if effect.target_type == EffectTargetType.CARD_TARGET:
            return True

    return False


class InvalidActionError(Exception):
    pass


def _handle_end_turn(cs: CombatState) -> list[ToBeQueuedEffect]:
    # Character's turn end
    to_be_queued_effects = get_end_of_turn_effects(
        cs.entity_manager, cs.entity_manager.id_character
    )

    # Monsters
    for id_monster in cs.entity_manager.id_monsters:
        monster = cs.entity_manager.entities[id_monster]

        # Turn start
        to_be_queued_effects += get_start_of_turn_effects(cs.entity_manager, id_monster)

        # Move's effects
        to_be_queued_effects += [
            ToBeQueuedEffect(effect, id_monster) for effect in monster.move_current.effects
        ]

        # Update move
        to_be_queued_effects += [
            ToBeQueuedEffect(Effect(EffectType.UPDATE_MOVE), id_target=id_monster)
        ]

        # Turn end
        to_be_queued_effects += get_end_of_turn_effects(cs.entity_manager, id_monster)

    # Character's turn start
    to_be_queued_effects += get_start_of_turn_effects(
        cs.entity_manager, cs.entity_manager.id_character
    )

    return to_be_queued_effects


def _handle_select_entity(cs: CombatState, id_target: int) -> list[ToBeQueuedEffect]:
    if cs.entity_manager.id_card_active is None and not cs.effect_queue:
        # Selected card
        card = cs.entity_manager.entities[id_target]
        if _card_requires_target(card):
            cs.entity_manager.id_card_active = id_target
            cs.entity_manager.id_selectables = cs.entity_manager.id_monsters

            return []

        # Play the card
        return [
            ToBeQueuedEffect(
                Effect(EffectType.PLAY_CARD), cs.entity_manager.id_character, id_target
            )
        ]

    if cs.entity_manager.id_card_active is not None and not cs.effect_queue:
        # Selected the active card's target, play the card
        id_card_active = cs.entity_manager.id_card_active

        cs.entity_manager.id_card_active = None
        cs.entity_manager.id_card_target = id_target

        return [
            ToBeQueuedEffect(
                Effect(EffectType.PLAY_CARD),
                cs.entity_manager.id_character,
                id_card_active,
            )
        ]

    if cs.entity_manager.id_card_active is None and cs.effect_queue:
        # Selected the active effect's target
        cs.entity_manager.id_effect_target = id_target

        return []


def handle_action(cs: CombatState, action: Action) -> list[ToBeQueuedEffect]:
    if action.type == ActionType.END_TURN:
        if cs.entity_manager.id_card_active is not None or cs.effect_queue:
            raise InvalidActionError

        return _handle_end_turn(cs)

    elif action.type == ActionType.SELECT_ENTITY:
        return _handle_select_entity(cs, action.target_id)


def step(cs: CombatState, action: Action) -> None:
    # Handle action
    to_be_queued_effects = handle_action(cs, action)
    for to_be_queued_effect in to_be_queued_effects:
        add_to_bot(
            cs.effect_queue,
            to_be_queued_effect.effect,
            to_be_queued_effect.id_source,
            to_be_queued_effect.id_target,
        )

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


def main(cs: CombatState, agent: BaseAgent) -> None:
    # Combat start
    to_be_queued_effects = get_start_of_combat_effects(cs.entity_manager)
    for to_be_queued_effect in to_be_queued_effects:
        add_to_bot(
            cs.effect_queue,
            to_be_queued_effect.effect,
            to_be_queued_effect.id_source,
            to_be_queued_effect.id_target,
        )
    # TODO: only call once
    process_effect_queue(cs.entity_manager, cs.effect_queue)

    # TODO: move
    cs.entity_manager.id_selectables = cs.entity_manager.id_cards_in_hand

    while not is_game_over(combat_state.entity_manager):
        # Get combat view and draw it on the terminal
        combat_view = view_combat(combat_state)
        draw_combat(combat_view)

        # Get action from agent
        action = agent.select_action(combat_view)

        # Game step
        step(cs, action)

    # TODO: combat end


if __name__ == "__main__":
    # Instance combat manager
    combat_state = create_combat_state()

    # Instance agent
    agent = RandomAgent()

    # Execute
    main(combat_state, agent)
