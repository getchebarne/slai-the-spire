from dataclasses import dataclass, replace

from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.entity.actor import ModifierType
from src.game.entity.character import EntityCharacter
from src.game.entity.manager import EntityManager
from src.game.entity.monster import EntityMonster
from src.game.view.actor import ViewActor
from src.game.view.actor import get_view_modifiers


@dataclass(frozen=True)
class Intent:
    damage: int | None
    instances: int | None
    block: bool
    buff: bool


@dataclass(frozen=True)
class ViewMonster(ViewActor):
    intent: Intent


def get_view_monsters(entity_manager: EntityManager) -> list[ViewMonster]:
    character = entity_manager.entities[entity_manager.id_character]

    monster_views = []
    for id_monster in entity_manager.id_monsters:
        monster = entity_manager.entities[id_monster]
        monster_views.append(_get_monster_view(monster, character))

    return monster_views


def _get_monster_view(monster: EntityMonster, character: EntityCharacter) -> ViewMonster:
    intent = _move_to_intent(monster.move_map[monster.move_name_current])
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


def _move_to_intent(move_effects: list[Effect]) -> Intent:
    # Initialze empty intent
    intent = Intent(None, None, False, False)

    # Iterate over the move's effects
    for effect in move_effects:
        if effect.type == EffectType.DAMAGE_DEAL:
            if intent.damage is None and intent.instances is None:
                intent = replace(intent, damage=effect.value, instances=1)

                continue

            if intent.damage != effect.value:
                raise ValueError("All of the move's damage instances must have the same value")

            intent = replace(intent, instances=intent.instances + 1)

        if not intent.block and effect.type == EffectType.BLOCK_GAIN:
            intent = replace(intent, block=True)

        # TODO: add support for other buffs
        if not intent.buff and (
            effect.type == EffectType.MODIFIER_STRENGTH_GAIN
            or effect.type == EffectType.MODIFIER_RITUAL_GAIN
        ):
            intent = replace(intent, buff=True)

    return intent


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
