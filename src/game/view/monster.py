from dataclasses import dataclass, replace

from src.game.entity.actor import ModifierType
from src.game.entity.character import EntityCharacter
from src.game.entity.manager import EntityManager
from src.game.entity.monster import EntityMonster
from src.game.entity.monster import Intent
from src.game.view.actor import ViewActor
from src.game.view.actor import get_view_modifiers


ViewIntent = Intent


@dataclass(frozen=True)
class ViewMonster(ViewActor):
    intent: Intent


def get_view_monsters(entity_manager: EntityManager) -> list[ViewMonster]:
    character = entity_manager.character

    monster_views = []
    for monster in entity_manager.monsters:
        monster_views.append(_get_monster_view(monster, character))

    return monster_views


def _get_monster_view(monster: EntityMonster, character: EntityCharacter) -> ViewMonster:
    intent = monster.moves[monster.move_name_current].intent
    if intent.damage is not None:
        damage_corrected = _get_corrected_intent_damage(intent.damage, monster, character)
        intent = replace(intent, damage=damage_corrected)

    return ViewMonster(
        monster.name,
        monster.health_current,
        monster.health_max,
        monster.block_current,
        get_view_modifiers(monster),
        intent,
    )


def _get_corrected_intent_damage(
    damage: int, monster: EntityMonster, character: EntityCharacter
) -> int:
    if ModifierType.STRENGTH in monster.modifier_map:
        damage += monster.modifier_map[ModifierType.STRENGTH].stacks_current

    if ModifierType.WEAK in monster.modifier_map:
        damage *= 0.75  # TODO: same as processor

    if ModifierType.VULNERABLE in character.modifier_map:
        damage *= 1.50  # TODO: same as processor

    return int(damage)
