from src.agents.random import RandomAgent
from src.game.battle.engine import BattleEngine
from src.game.context import Context
from src.game.context import EntityData


if __name__ == "__main__":
    # Instantiate the agent
    agent = RandomAgent()

    # Instantiate the context
    entities = {
        0: EntityData(name="Silent", max_health=50),
        1: EntityData(name="Dummy", max_health=50),
    }
    context = Context(entities=entities)

    # Instantiate the game engine
    engine = BattleEngine(agent, context)

    engine.run()
