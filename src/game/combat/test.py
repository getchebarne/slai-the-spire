from src.agents.random import RandomAgent
from src.game.combat.action import ActionType
from src.game.combat.context import GameContext
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
from src.game.combat.phase import turn_start_character
from src.game.combat.view import view_combat


context = GameContext(
    character=silent(),
    monsters=[dummy()],
    energy=energy(),
    deck={
        strike(),
        strike(),
        strike(),
        strike(),
        strike(),
        defend(),
        defend(),
        defend(),
        defend(),
        defend(),
    },
)
agent = RandomAgent()


def main():
    combat_start(context)

    while not is_game_over(context):
        turn_start_character(context)
        action = agent.select_action(context)

        while action.type != ActionType.END_TURN:
            print(action)
            combat_view = view_combat(context)
            drawer(combat_view)
            character_turn(context, action)
            action = agent.select_action(context)

        turn_end_character(context)

        # Monster turn


if __name__ == "__main__":
    main()
