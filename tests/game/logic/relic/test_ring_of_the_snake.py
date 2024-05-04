from src.agents.random import RandomAgent
from src.game.battle.engine import BattleEngine
from src.game.context import Context
from src.game.context import EntityData
from src.game.logic.relic.ring_of_the_snake import DRAW


MONSTER_ENTITY_ID = 1


def test_base():
    # Instantiate the Agent, Context, and Engine
    agent = RandomAgent()
    context = Context(
        entities={
            Context.CHAR_ENTITY_ID: EntityData(name="Silent", max_health=50),
            MONSTER_ENTITY_ID: EntityData(name="Dummy", max_health=50),
        },
        relics=["Ring of the Snake"],  # TODO: make relic names an enum?
    )
    engine = BattleEngine(agent, context)

    # Start the battle
    engine.battle_start()

    # Assert that there's DRAW cards in the hand
    assert len(context.hand) == DRAW
