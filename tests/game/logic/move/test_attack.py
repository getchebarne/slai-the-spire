from src.agents.random import RandomAgent
from src.game.battle.engine import BattleEngine
from src.game.context import Context
from src.game.context import EntityData
from src.game.logic.move.attack import DAMAGE


MONSTER_ENTITY_ID = 1


def test_base():
    # Instantiate the Agent, Context, and Engine
    agent = RandomAgent()
    context = Context(
        entities={
            Context.CHAR_ENTITY_ID: EntityData(name="Silent", max_health=50),
            MONSTER_ENTITY_ID: EntityData(name="Dummy", max_health=50),
        },
        monster_moves={MONSTER_ENTITY_ID: "Attack"},
    )
    engine = BattleEngine(agent, context)

    # Store previous health
    prev_health = context.entities[context.CHAR_ENTITY_ID].current_health

    # Execute the move
    engine._execute_monster_move(MONSTER_ENTITY_ID)

    # Assert that the character's health has decreased by DAMAGE
    assert context.entities[context.CHAR_ENTITY_ID].current_health == prev_health - DAMAGE
