from src.agents.random import RandomAgent
from src.game.combat.create import create_combat
from src.game.combat.drawer import drawer
from src.game.combat.effect_queue import EffectQueue
from src.game.combat.handle_input import handle_action
from src.game.combat.phase import combat_start
from src.game.combat.phase import turn_monster
from src.game.combat.state import GameState
from src.game.combat.utils import is_game_over
from src.game.combat.view import view_combat


# Create state
state = GameState()

# Instance state & effect queue
state = create_combat()
effect_queue = EffectQueue()

combat_start(state, effect_queue)

# Instance agent
agent = RandomAgent()


def main():
    # Game loop
    while not is_game_over(state):
        #
        combat_view = view_combat(state, effect_queue)
        drawer(combat_view)

        # Get action form agent
        if state.actor_turn_id == state.character_id:
            action = agent.select_action(combat_view)

            # Handle action
            handle_action(state, effect_queue, action)

        # Monsters turn
        turn_monster(state, effect_queue)


if __name__ == "__main__":
    main()
