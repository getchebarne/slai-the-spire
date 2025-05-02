from dataclasses import dataclass

from src.game.entity.manager import EntityManager
from src.game.view.actor import ViewActor
from src.game.view.actor import get_view_modifiers


@dataclass(frozen=True)
class ViewCharacter(ViewActor):
    card_reward_roll_offset: int


def get_view_character(entity_manager: EntityManager) -> ViewCharacter:
    character = entity_manager.entities[entity_manager.id_character]

    return ViewCharacter(
        character.name,
        character.health_current,
        character.health_max,
        character.block_current,
        get_view_modifiers(character),
        character.card_reward_roll_offset,
    )
