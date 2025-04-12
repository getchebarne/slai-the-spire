from typing import Callable

from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.drawer import draw_combat
from src.game.combat.effect import Effect
from src.game.combat.effect import EffectType
from src.game.combat.effect import SourcedEffect
from src.game.combat.effect_queue import add_to_bot
from src.game.combat.effect_queue import add_to_top
from src.game.combat.effect_queue import process_effect_queue
from src.game.combat.phase import get_start_of_combat_effects
from src.game.combat.state import CombatState
from src.game.combat.state import FSMState
from src.game.combat.utils import does_card_require_target
from src.game.combat.utils import is_game_over
from src.game.combat.view import CombatView
from src.game.combat.view import view_combat


class InvalidActionError(Exception):
    pass


class InvalidStateError(Exception):
    pass


def _handle_select_entity(
    cs: CombatState, id_target: int
) -> tuple[list[SourcedEffect], list[SourcedEffect]]:
    if cs.fsm_state == FSMState.DEFAULT:
        if id_target not in cs.entity_manager.id_cards_in_hand:
            raise InvalidActionError("Can only select cards in hand while in default state")

        # Get the selected card
        card = cs.entity_manager.entities[id_target]

        # Check if there's enough energy to play it
        energy_current = cs.entity_manager.entities[cs.entity_manager.id_energy].current
        if card.cost > energy_current:
            raise InvalidActionError(f"Can't select card {card} with {energy_current} energy")

        # If the card requires targeting, set it as active and return
        if does_card_require_target(card):
            return (
                [SourcedEffect(Effect(EffectType.CARD_ACTIVE_SET), id_target=id_target)],
                [],
            )

        # Else, play it right away. This is different from the original game's implementation,
        # where cards that don't need a target are still set as active and await the player's
        # confirmation
        return (
            [SourcedEffect(Effect(EffectType.PLAY_CARD), id_target=id_target)],
            [],
        )

    if cs.fsm_state == FSMState.AWAIT_TARGET_CARD:
        # Queue is empty in this state. First the card's target is set, then the active card is
        # cleared, then the card is played (it's effects are added to the top of the queue), and
        # finally the card's target is cleared
        return (
            [
                SourcedEffect(Effect(EffectType.TARGET_CARD_SET), id_target=id_target),
                SourcedEffect(Effect(EffectType.CARD_ACTIVE_CLEAR)),
                SourcedEffect(
                    Effect(EffectType.PLAY_CARD), id_target=cs.entity_manager.id_card_active
                ),
                SourcedEffect(Effect(EffectType.TARGET_CARD_CLEAR)),
            ],
            [],
        )

    if cs.fsm_state == FSMState.AWAIT_TARGET_EFFECT:
        # An effect is added at the top of the queue to set the effect's target. The effect that
        # comes inmediately after is going to use this variable to resolve its target. For now,
        # an effect to clear the effect's target is added to the bottom of the queue, but TODO:
        # I think I can't escape adding the clear when the `id_effect_target` is consumed
        return (
            [SourcedEffect(Effect(EffectType.TARGET_EFFECT_CLEAR), id_target=id_target)],
            [SourcedEffect(Effect(EffectType.TARGET_EFFECT_SET), id_target=id_target)],
        )


def handle_action(
    cs: CombatState, action: Action
) -> tuple[list[SourcedEffect], list[SourcedEffect]]:
    if action.type == ActionType.END_TURN:
        return (
            [SourcedEffect(Effect(EffectType.END_TURN))],
            [],
        )

    elif action.type == ActionType.SELECT_ENTITY:
        return _handle_select_entity(cs, action.target_id)


def step(cs: CombatState, action: Action) -> None:
    # Handle action
    sourced_effects_bot, sourced_effects_top = handle_action(cs, action)

    # Add new effects to the queue
    add_to_bot(cs.effect_queue, *sourced_effects_bot)
    add_to_top(cs.effect_queue, *sourced_effects_top)

    # Process round
    process_effect_queue(cs.entity_manager, cs.effect_queue)

    # Set new state
    _set_new_state(cs)


def _set_new_state(cs: CombatState) -> None:
    if cs.entity_manager.id_card_active is not None:
        if cs.effect_queue:
            raise InvalidStateError(
                f"Can't enter {FSMState.AWAIT_TARGET_CARD.name} state with non-empty effect queue"
            )

        cs.fsm_state = FSMState.AWAIT_TARGET_CARD

        return

    if cs.effect_queue:
        cs.fsm_state = FSMState.AWAIT_TARGET_EFFECT

        return

    cs.fsm_state = FSMState.DEFAULT

    return


def start_combat(cs: CombatState) -> None:
    # Queue start of combat effects
    sourced_effects = get_start_of_combat_effects(cs.entity_manager)
    add_to_bot(cs.effect_queue, *sourced_effects)

    # Process them
    process_effect_queue(cs.entity_manager, cs.effect_queue)

    # Set new state
    _set_new_state(cs)


def main(cs: CombatState, select_action_fn: Callable[[CombatView], Action]) -> None:
    start_combat(cs)

    while not is_game_over(cs.entity_manager):
        # Get combat view and draw it on the terminal
        combat_view = view_combat(cs)
        draw_combat(combat_view)

        # Get action from agent
        action, _ = select_action_fn(combat_view)

        # Game step
        step(cs, action)

    # TODO: combat end


if __name__ == "__main__":
    from src.game.combat.create import create_combat_state
    from src.rl.policies import PolicyRandom

    cs = create_combat_state()

    main(cs, PolicyRandom().select_action)
