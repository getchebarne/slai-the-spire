from enum import Enum
from dataclasses import dataclass
from typing import Optional, Generator

from src.game.core.energy import Energy


# TODO: define elsewhere
class BattleState(Enum):
    DEFAULT = 0
    AWAIT_TARGET = 1
    NONE = 2


@dataclass
class EntityData:
    name: str
    max_health: int
    current_health: Optional[int] = None
    current_block: int = 0

    def __post_init__(self):
        if self.current_health is None:
            self.current_health = self.max_health


class Context:
    CHAR_ENTITY_ID = 0

    def __init__(
        self,
        entities: dict[int, EntityData],
        entity_modifiers: dict[tuple[int, str], int] = {},
        monster_moves: dict[int, Optional[str]] = {},
        deck: list[str] = [],
        hand: list[str] = [],
        draw_pile: list[str] = [],
        disc_pile: list[str] = [],
        active_card_idx: Optional[int] = None,
        energy: Energy = Energy(max=3, current=3),
        state: BattleState = BattleState.NONE,
    ):
        self.entities = entities
        self.entity_modifiers = entity_modifiers
        self.monster_moves = monster_moves
        self.deck = deck
        self.hand = hand
        self.draw_pile = draw_pile
        self.disc_pile = disc_pile
        self.active_card_idx = active_card_idx
        self.energy = energy
        self.state = state

        # Setup
        self._setup()

    def _setup(self) -> None:
        # Initialize monster moves. TODO: fix
        if self.monster_moves == {}:
            self.monster_moves = {
                entity_id: None
                for entity_id in self.entities
                if entity_id != Context.CHAR_ENTITY_ID
            }
        # Initialize deck. TODO: use starter deck from database
        if self.deck == []:
            self.deck = [
                "Strike",
                "Strike",
                "Strike",
                "Strike",
                "Strike",
                "Defend",
                "Defend",
                "Defend",
                "Defend",
                "Defend",
                "Neutralize",
            ]

    def get_monsters(self) -> Generator[tuple[int, EntityData], None, None]:
        for entity_id, entity_data in self.entities.items():
            if entity_id != Context.CHAR_ENTITY_ID:
                yield entity_id, entity_data

    def get_char(self) -> tuple[int, EntityData]:
        return Context.CHAR_ENTITY_ID, self.entities[Context.CHAR_ENTITY_ID]
