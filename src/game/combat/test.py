from src.agents.random import RandomAgent
from src.game.combat.create import create_combat
from src.game.combat.drawer import drawer
from src.game.combat.effect_queue import EffectQueue
from src.game.combat.effect_queue import process_queue
from src.game.combat.entities import Entities
from src.game.combat.handle_input import handle_action
from src.game.combat.phase import combat_start
from src.game.combat.state import State
from src.game.combat.utils import is_game_over
from src.game.combat.view import view_combat


# Create entities
entities = Entities()

# Instance entities & effect queue
entities = create_combat()
effect_queue = EffectQueue()

combat_start(entities, effect_queue)

# Instance agent
agent = RandomAgent()


def main():
    # Game loop
    state = State.DEFAULT  # TODO: should be done by combat_start

    while not is_game_over(entities):
        #
        combat_view = view_combat(entities, effect_queue, state)
        print(state)
        drawer(combat_view)

        # Get action form agent
        action = agent.select_action(combat_view)
        print(action)

        # Handle action
        state = handle_action(entities, effect_queue, state, action)

        # Process effect queue
        process_queue(entities, effect_queue)

        if effect_queue.get_pending() is None:
            continue

        state = State.AWAIT_EFFECT_TARGET


if __name__ == "__main__":
    main()
