from src.agents.random import RandomAgent
from src.game.combat.action import ActionType
from src.game.combat.state import GameState
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
from src.game.combat.view import view_combat


context = GameState(
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
    # Start combat
    combat_start(context)

    # Game loop
    while not is_game_over(context):
        # Character turn start
        turn_start_character(context)

        # Character turn
        action = None
        while action is None or action.type != ActionType.END_TURN:
            combat_view = view_combat(context)
            drawer(combat_view)
            action = agent.select_action(combat_view)
            character_turn(context, action)

        # Character turn end
        turn_end_character(context)

        # Monsters turn
        for monster in context.monsters:
            # Monster turn start
            turn_start_monster(context, monster)

            # Monster turn
            turn_monster(context, monster)

            # Monster turn end
            turn_end_monster(context, monster)


if __name__ == "__main__":
    main()
