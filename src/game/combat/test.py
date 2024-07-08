from src.agents.random import RandomAgent
from src.game.combat.create import create_combat
from src.game.combat.drawer import drawer
from src.game.combat.effect_queue import EffectQueue
from src.game.combat.entities import Entities
from src.game.combat.handle_input import handle_action
from src.game.combat.phase import combat_start
from src.game.combat.phase import turn_monster
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
    while not is_game_over(entities):
        #
        combat_view = view_combat(entities, effect_queue)
        drawer(combat_view)

        # Get action form agent
        if entities.actor_turn_id == entities.character_id:
            action = agent.select_action(combat_view)

            # Handle action
            handle_action(entities, effect_queue, action)

        # Monsters turn
        turn_monster(entities, effect_queue)


if __name__ == "__main__":
    main()
