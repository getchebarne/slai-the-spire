from src.agents.random import RandomAgent
from src.game.combat.drawer import drawer
from src.game.combat.factories import defend
from src.game.combat.factories import dummy
from src.game.combat.factories import energy
from src.game.combat.factories import neutralize
from src.game.combat.factories import silent
from src.game.combat.factories import strike
from src.game.combat.factories import survivor
from src.game.combat.handle_input import handle_action
from src.game.combat.phase import combat_start
from src.game.combat.phase import turn_monster
from src.game.combat.state import GameState
from src.game.combat.utils import is_game_over
from src.game.combat.view import view_combat


# Create state
state = GameState()

# Fill state
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
    state.create_entity(survivor()),
    state.create_entity(neutralize()),
}

# Instance agent
agent = RandomAgent()


def main():
    # Start combat
    combat_start(state)

    # Game loop
    while not is_game_over(state):
        #
        combat_view = view_combat(state)
        drawer(combat_view)

        # Get action form agent
        if state.actor_turn_id == state.character_id:
            action = agent.select_action(combat_view)

            # Handle action
            handle_action(state, action)

        # Monsters turn
        turn_monster(state)


if __name__ == "__main__":
    main()
