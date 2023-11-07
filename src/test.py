from agents.random import RandomAgent
from game.battle.context import BattleContext
from game.battle.drawer import BattleDrawer
from game.battle.engine import BattleEngine
from game.entities.actors.base import Health
from game.entities.actors.characters.base import Character
from game.entities.actors.monsters.dummy import Dummy
from game.entities.actors.monsters.group import MonsterGroup
from game.entities.cards.deck import SILENT_STARTER_DECK
from game.entities.cards.disc_pile import DiscardPile
from game.entities.cards.draw_pile import DrawPile
from game.entities.cards.hand import Hand
from game.entities.relics.base import Relics
from game.entities.relics.vajra import Vajra


if __name__ == "__main__":
    # Instance Agent
    agent = RandomAgent()

    relics = Relics()

    # Instance battle context
    context = BattleContext(
        char=Character(health=Health(10)),
        monsters=MonsterGroup([Dummy()]),
        deck=SILENT_STARTER_DECK,
        disc_pile=DiscardPile([]),
        draw_pile=DrawPile(SILENT_STARTER_DECK.cards),
        hand=Hand([]),
        relics=relics,
    )
    relics.add_relic(Vajra(), context.pipeline)

    # Instance drawer
    drawer = BattleDrawer()

    # Instance Battle
    engine = BattleEngine(agent, context, drawer)

    # Start
    engine.run()
