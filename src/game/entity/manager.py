from dataclasses import dataclass, field

from src.game.entity.base import EntityBase


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


def add_entity(entity_manager: EntityManager, entity: EntityBase) -> int:
    id_ = _get_first_free_id(entity_manager)
    entity_manager.entities[id_] = entity

    return id_


def delete_entity(entity_manager: EntityManager, id_: int) -> None:
    entity_manager.entities.pop(id_)


def _get_first_free_id(entity_manager: EntityManager) -> int:
    id_used = set(entity_manager.entities.keys())

    id_ = 0
    while id_ in id_used:
        id_ += 1

    return id_
