from dataclasses import dataclass, field

from src.game.entity.base import EntityBase


@dataclass
class EntityManager:
    entities: list[EntityBase]

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
    id_map_nodes: list[list[int | None]] = field(default_factory=list)
    id_map_node_active: int | None = None
    id_map_node_boss: int | None = None

    id_card_target: int | None = None


def create_entity(entity_manager: EntityManager, entity: EntityBase) -> int:
    entity_manager.entities.append(entity)

    # Return the entity's index
    return len(entity_manager.entities) - 1
