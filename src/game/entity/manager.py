from dataclasses import dataclass, field

from src.game.entity.base import EntityBase


@dataclass
class EntityManager:
    entities: list[EntityBase]

    id_character: int | None = None
    id_monsters: list[int] | None = None
    id_energy: int | None = None
    id_cards_in_deck: list[int] | None = None
    id_cards_in_draw_pile: list[int] = field(default_factory=list)
    id_cards_in_hand: list[int] = field(default_factory=list)
    id_cards_in_disc_pile: list[int] = field(default_factory=list)
    id_card_active: int | None = None

    # TODO: can maybe fuse into one single `id_target`?
    id_card_target: int | None = None
    id_effect_target: int | None = None

    id_selectables: list[int] | None = None


def create_entity(entity_manager: EntityManager, entitiy: EntityBase) -> int:
    entity_manager.entities.append(entitiy)

    # Return the entity's index
    return len(entity_manager.entities) - 1
