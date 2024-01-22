from agents.random import RandomAgent
from game.battle.engine import BattleEngine


if __name__ == "__main__":
    agent = RandomAgent()
    engine = BattleEngine(agent)

    engine.run()
