from src.game.combat.factories import defend
from src.game.combat.factories import dummy
from src.game.combat.factories import energy
from src.game.combat.factories import silent
from src.game.combat.factories import strike
from src.game.combat.phase import combat_start
from src.game.combat.state import GameState


def create_combat() -> GameState:
    # Create GameState instance
    state = GameState()

    # Fill
    # TODO: create functions for this
    state.character_id = state.create_entity(silent())
    state.monster_ids = [state.create_entity(dummy())]
    state.energy_id = state.create_entity(energy())
    state.card_in_deck_ids = {
        state.create_entity(strike()),
        state.create_entity(strike()),
        state.create_entity(strike()),
        state.create_entity(strike()),
        state.create_entity(strike()),
        state.create_entity(defend()),
        state.create_entity(defend()),
        state.create_entity(defend()),
        state.create_entity(defend()),
        state.create_entity(defend()),
        # state.create_entity(survivor()),
        # state.create_entity(neutralize()),
    }

    # Trigger combat start
    combat_start(state)

    return state
