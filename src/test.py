from agents.random import RandomAgent
from game.battle.engine import BattleEngine
from game.context import Context
from game.context import EntityData

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
