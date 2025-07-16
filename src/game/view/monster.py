from dataclasses import dataclass, replace

from src.game.entity.character import EntityCharacter
from src.game.entity.manager import EntityManager
from src.game.entity.monster import EntityMonster
from src.game.entity.monster import Intent
from src.game.utils import get_corrected_intent_damage
from src.game.view.actor import ViewActor
from src.game.view.actor import get_view_modifiers


ViewIntent = Intent


@dataclass(frozen=True)
class ViewMonster(ViewActor):
    intent: Intent


def get_view_monsters(entity_manager: EntityManager) -> list[ViewMonster]:
    character = entity_manager.character

    monster_views = []
    for id_monster in entity_manager.id_monsters:
        monster = entity_manager.entities[id_monster]
        monster_views.append(_get_monster_view(monster, character))

    return monster_views


def _get_monster_view(monster: EntityMonster, character: EntityCharacter) -> ViewMonster:
    intent = monster.moves[monster.move_name_current].intent
    if intent.damage is not None:
        damage_corrected = get_corrected_intent_damage(intent.damage, monster, character)
        intent = replace(intent, damage=damage_corrected)

    return ViewMonster(
        monster.name,
        monster.health_current,
        monster.health_max,
        monster.block_current,
        get_view_modifiers(monster),
        intent,
    )
