from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.effect_queue import process_queue
from src.game.combat.state import GameState
from src.game.combat.utils import add_effects_to_bot
from src.game.combat.utils import card_requires_target


# TODO: should be EffectPlayCard
def play_card(state: GameState) -> None:
    active_card = state.get_active_card()
    energy = state.get_energy()
    if active_card.cost > energy.current:
        raise ValueError(f"Can't play card {active_card} with {energy.current} energy")

    # Subtract energy
    energy.current -= active_card.cost

    # Send card to discard pile
    state.card_in_hand_ids.remove(state.card_active_id)
    state.card_in_discard_pile_ids.add(state.card_active_id)


def is_game_over(state: GameState) -> bool:
    return state.get_character().health.current <= 0 or all(
        [monster.health.current <= 0 for monster in state.get_monsters()]
    )


def character_turn(state: GameState, action: Action) -> None:
    if action.type == ActionType.SELECT_CARD:
        state.card_active_id = state.card_in_hand_ids[action.index]
        if card_requires_target(state.get_active_card()):
            # Wait for player input
            return

    if action.type == ActionType.SELECT_MONSTER:
        state.card_target_id = state.monster_ids[action.index]

    if action.type == ActionType.END_TURN:
        return

    # Play card. TODO: confirm?
    play_card(state)
    add_effects_to_bot(state, *state.get_active_card().effects)
    process_queue(state)

    # Untag card
    state.card_active_id = None
