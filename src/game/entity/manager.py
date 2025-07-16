from dataclasses import dataclass, field

from src.game.entity.base import EntityBase
from src.game.entity.card import EntityCard
from src.game.entity.character import EntityCharacter
from src.game.entity.energy import EntityEnergy
from src.game.entity.map_node import EntityMapNode
from src.game.entity.monster import EntityMonster


@dataclass
class EntityManager:
    entities: dict[int, EntityBase]

    id_character: int | None = None
    id_monsters: list[int] = field(default_factory=list)
    id_energy: int | None = None
    id_cards_in_deck: list[int] = field(default_factory=list)
    id_cards_in_draw_pile: list[int] = field(default_factory=list)
    id_cards_in_hand: list[int] = field(default_factory=list)
    id_cards_in_disc_pile: list[int] = field(default_factory=list)
    id_cards_in_exhaust_pile: list[int] = field(default_factory=list)
    id_card_reward: list[int] = field(default_factory=list)
    id_card_active: int | None = None

    # Map
    id_map_nodes: list[list[int | None]] = field(default_factory=dict)
    id_map_node_active: int | None = None
    id_map_node_boss: int | None = None

    # Card target
    id_card_target: int | None = None

    @property
    def character(self) -> EntityCharacter:
        return self.entities[self.id_character]

    @property
    def monsters(self) -> list[EntityMonster]:
        return [self.entities[id_] for id_ in self.id_monsters]

    @property
    def energy(self) -> EntityEnergy:
        return self.entities[self.id_energy]

    @property
    def cards_in_deck(self) -> list[EntityCard]:
        return [self.entities[id_] for id_ in self.id_cards_in_deck]

    @property
    def cards_in_draw_pile(self) -> list[EntityCard]:
        return [self.entities[id_] for id_ in self.id_cards_in_draw_pile]

    @property
    def cards_in_hand(self) -> list[EntityCard]:
        return [self.entities[id_] for id_ in self.id_cards_in_hand]

    @property
    def cards_in_disc_pile(self) -> list[EntityCard]:
        return [self.entities[id_] for id_ in self.id_cards_in_disc_pile]

    @property
    def cards_in_exhaust_pile(self) -> list[EntityCard]:
        return [self.entities[id_] for id_ in self.id_cards_in_exhaust_pile]

    @property
    def cards_reward(self) -> list[EntityCard]:
        return [self.entities[id_] for id_ in self.id_card_reward]

    @property
    def card_active(self) -> EntityCard:
        return self.entities[self.id_card_active]

    @property
    def map_nodes(self) -> list[list[EntityMapNode | None]]:
        return [
            [None if id_ is None else self.entities[id_] for id_ in row]
            for row in self.id_map_nodes
        ]

    @property
    def map_node_active(self) -> EntityMapNode | None:
        if self.id_map_node_active is None:
            return None

        return self.entities[self.id_map_node_active]

    @property
    def map_node_boss(self) -> EntityMapNode:
        return self.entities[self.id_map_node_boss]

    @property
    def card_target(self) -> EntityMonster:
        return self.entities[self.id_card_target]


def add_entity(entity_manager: EntityManager, entity: EntityBase) -> int:
    id_ = _get_first_free_id(entity_manager)
    entity_manager.entities[id_] = entity

    return id_


def delete_entity(entity_manager: EntityManager, id_: int) -> None:
    entity_manager.entities.pop(id_)


def get_entity(entity_manager: EntityManager, id_: int) -> EntityBase:
    return entity_manager.entities[id_]


def _get_first_free_id(entity_manager: EntityManager) -> int:
    id_used = set(entity_manager.entities.keys())

    id_ = 0
    while id_ in id_used:
        id_ += 1

    return id_
