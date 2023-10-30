from dataclasses import dataclass
from enum import Enum

from game.battle.state import BattleState
from game.entities.actors.char import Character
from game.entities.actors.monster import MonsterCollection
from game.entities.cards.disc_pile import DiscardPile
from game.entities.cards.draw_pile import DrawPile
from game.entities.cards.hand import Hand


class ActionType(Enum):
    PLAY_CARD = 0
    SELECT_TARGET = 1
    END_TURN = 2


@dataclass
class BattleView:
    state: BattleState
    char: Character
    monsters: MonsterCollection
    disc_pile: DiscardPile
    draw_pile: DrawPile
    hand: Hand
