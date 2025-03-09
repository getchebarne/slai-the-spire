from dataclasses import dataclass

from src.game.combat.entities import Effect
from src.game.combat.entities import EffectType
from src.game.combat.entities import Entities
from src.game.combat.entities import Monster
from src.game.combat.view.actor import ActorView
from src.game.combat.view.actor import _actor_to_view


@dataclass
class IntentView:
    damage: int | None
    instances: int | None
    block: bool
    buff: bool


@dataclass
class MonsterView(ActorView):
    entity_id: int
    intent: IntentView


def _move_effects_to_intent(move_effects: list[Effect]) -> IntentView:
    # Initialze empty intent
    intent = IntentView(None, None, False, False)

    # Iterate over the move's effects
    for effect in move_effects:
        if effect.type == EffectType.DEAL_DAMAGE:
            if intent.damage is None and intent.instances is None:
                intent.damage = effect.value
                intent.instances = 1

                continue

            if intent.damage != effect.value:
                raise ValueError("All of the move's damage instances must have the same value")

            intent.instances += 1

        if not intent.block and effect.type == EffectType.GAIN_BLOCK:
            intent.block = True

        # TODO: add support for other buffs
        if not intent.buff and effect.type == EffectType.GAIN_STR:
            intent.buff = True

    return intent


# TODO: should only be calculated in 1 place
# TODO: reenable
# def _correct_intent_damage(damage: int | None, monster: Monster) -> int | None:
#     if damage is None:
#         return

#     if ModifierType.STR in monster.modifiers:
#         damage += monster.modifiers[ModifierType.STR].stacks

#     if ModifierType.WEAK in monster.modifiers:
#         damage *= 0.75

#     return int(damage)


def _monster_to_view(entities: Entities, monster_entity_id: int) -> MonsterView:
    monster = entities.all[monster_entity_id]
    actor_view = _actor_to_view(monster)
    intent_view = _move_effects_to_intent(monster.move_current.effects)
    # intent_view.damage = _correct_intent_damage(intent_view.damage, monster)

    return MonsterView(
        actor_view.name,
        actor_view.health_current,
        actor_view.health_max,
        actor_view.block_current,
        # actor_view.modifiers,
        monster_entity_id,  # TODO: revisit order
        intent_view,
    )


def view_monsters(entities: Entities) -> list[MonsterView]:
    return [
        _monster_to_view(entities, monster_entity_id) for monster_entity_id in entities.monster_ids
    ]
