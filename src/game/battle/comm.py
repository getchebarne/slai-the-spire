from dataclasses import dataclass

from game.battle.state import BattleState
from game.entities.actors.char import Character
from game.entities.actors.monster import MonsterCollection
from game.entities.cards.disc_pile import DiscardPile
from game.entities.cards.draw_pile import DrawPile
from game.entities.cards.hand import Hand


@dataclass
class BattleView:
    state: BattleState
    char: Character
    monsters: MonsterCollection
    disc_pile: DiscardPile
    draw_pile: DrawPile
    hand: Hand
