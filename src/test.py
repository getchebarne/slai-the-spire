from agents.random import RandomAgent
from game.battle.context import BattleContext
from game.battle.drawer import BattleDrawer
from game.battle.engine import BattleEngine
from game.entities.actors.base import Health
from game.entities.actors.char import Character
from game.entities.actors.dummy import Dummy
from game.entities.actors.monster import MonsterCollection
from game.entities.cards.deck import SILENT_STARTER_DECK
from game.entities.cards.disc_pile import DiscardPile
from game.entities.cards.draw_pile import DrawPile
from game.entities.cards.hand import Hand


if __name__ == "__main__":
    # Instance Agent
    agent = RandomAgent()

    # Instance battle context
    context = BattleContext(
        char=Character(health=Health(10)),
        monsters=MonsterCollection([Dummy()]),
        deck=SILENT_STARTER_DECK,
        disc_pile=DiscardPile([]),
        draw_pile=DrawPile(SILENT_STARTER_DECK.cards),
        hand=Hand([]),
    )
    # Instance drawer
    drawer = BattleDrawer()

    # Instance Battle
    engine = BattleEngine(agent, context, drawer)

    # Start
    engine.run()
