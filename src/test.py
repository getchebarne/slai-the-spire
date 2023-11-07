from agents.random import RandomAgent
from game.battle.context import BattleContext
from game.battle.engine import BattleEngine
from game.entities.actors.base import Health
from game.entities.actors.characters.base import Character
from game.entities.actors.monsters.dummy import Dummy
from game.entities.actors.monsters.group import MonsterGroup
from game.entities.cards.deck import SILENT_STARTER_DECK
from game.entities.relics.group import RelicGroup
from game.entities.relics.vajra import Vajra

if __name__ == "__main__":
    # Instance Agent
    agent = RandomAgent()

    # Instance Relics
    relics = RelicGroup([Vajra()])

    # Instance battle context
    context = BattleContext(
        char=Character(health=Health(10)),
        monsters=MonsterGroup([Dummy()]),
        deck=SILENT_STARTER_DECK,
        relics=relics,
    )
    # Instance Battle
    engine = BattleEngine(agent, context)

    # Start
    engine.run()
