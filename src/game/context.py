from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Generator, Optional

from src.game.core.energy import Energy


STARTING_ENERGY = 3


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
        entity_modifiers: Optional[dict[tuple[int, str], int]] = None,
        monster_moves: Optional[dict[int, Optional[str]]] = None,
        relics: Optional[list[str]] = None,
        deck: Optional[list[str]] = None,
        hand: Optional[list[str]] = None,
        draw_pile: Optional[list[str]] = None,
        disc_pile: Optional[list[str]] = None,
        active_card_idx: Optional[int] = None,
        energy: Optional[Energy] = None,
        state: BattleState = BattleState.NONE,
    ):
        self.entities = entities

        # Initialize optional parameters
        self.entity_modifiers = entity_modifiers or {}
        self.monster_moves = monster_moves or {}
        self.relics = relics or []
        self.deck = deck or []
        self.hand = hand or []
        self.draw_pile = draw_pile or []
        self.disc_pile = disc_pile or []
        self.active_card_idx = active_card_idx
        self.energy = energy or Energy(max=STARTING_ENERGY)
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

        # Transform entity_modifiers into a defaultdict
        self.entity_modifiers = defaultdict(lambda: 0, self.entity_modifiers)

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
