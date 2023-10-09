from dataclasses import dataclass
from typing import Literal

from components.block import Block
from components.energy import Energy
from components.health import Health
from entities.base import BaseEntity
from entities.deck import Deck
from entities.disc_pile import DiscardPile
from entities.draw_pile import DrawPile
from entities.hand import Hand


ALL_CHARS = Literal["Ironclad", "Silent", "Defect", "Watcher"]


@dataclass
class Player(BaseEntity):
    block: Block
    deck: Deck
    disc_pile: DiscardPile
    draw_pile: DrawPile
    energy: Energy
    hand: Hand
    health: Health
    name: ALL_CHARS
