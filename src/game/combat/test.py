from src.agents.random import RandomAgent
from src.game.combat.create import create_combat
from src.game.combat.drawer import drawer
from src.game.combat.handle_input import handle_action
from src.game.combat.manager import CombatManager
from src.game.combat.phase import combat_start
from src.game.combat.utils import is_game_over
from src.game.combat.view import view_combat


# Instantiate combat manager
combat_manager = CombatManager(entities=create_combat())
combat_start(combat_manager)

# Instance agent
agent = RandomAgent()


def main():
    while not is_game_over(combat_manager.entities):
        # Get combat view and draw it on the terminal
        combat_view = view_combat(combat_manager)
        drawer(combat_view)

        # Get action form agent
        action = agent.select_action(combat_view)

        # Handle action
        handle_action(combat_manager, action)


if __name__ == "__main__":
    main()
