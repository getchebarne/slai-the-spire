from src.agents.random import RandomAgent
from src.game.combat.action import ActionType
from src.game.combat.drawer import drawer
from src.game.combat.factories import defend
from src.game.combat.factories import dummy
from src.game.combat.factories import energy
from src.game.combat.factories import silent
from src.game.combat.factories import strike
from src.game.combat.logic import character_turn
from src.game.combat.logic import is_game_over
from src.game.combat.phase import combat_start
from src.game.combat.phase import turn_end_character
from src.game.combat.phase import turn_end_monster
from src.game.combat.phase import turn_monster
from src.game.combat.phase import turn_start_character
from src.game.combat.phase import turn_start_monster
from src.game.combat.state import GameState
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
}

# Instance agent
agent = RandomAgent()


def main():
    # Start combat
    combat_start(state)

    # Game loop
    while not is_game_over(state):
        # Character turn start
        turn_start_character(state)

        # Character turn. TODO: fix gamer over
        action = None
        while action is None or action.type != ActionType.END_TURN:
            combat_view = view_combat(state)
            drawer(combat_view)
            action = agent.select_action(combat_view)
            character_turn(state, action)

        # Character turn end
        turn_end_character(state)

        # Monsters turn
        for monster_id in state.monster_ids:
            # Monster turn start
            turn_start_monster(state, monster_id)

            # Monster turn
            turn_monster(state, monster_id)

            # Monster turn end
            turn_end_monster(state, monster_id)


if __name__ == "__main__":
    main()
