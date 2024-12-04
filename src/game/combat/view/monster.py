from dataclasses import dataclass

from src.game.combat.entities import EffectType
from src.game.combat.entities import Entities
from src.game.combat.entities import MonsterMove
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


def _move_to_intent(move: MonsterMove) -> IntentView:
    # Initialze empty intent
    intent = IntentView(None, None, False, False)

    # Iterate over the move's effects
    for effect in move.effects:
        if effect.type == EffectType.DEAL_DAMAGE:
            if intent.damage is None and intent.instances is None:
                intent.damage = effect.value
                intent.instances = 1

                continue

            if intent.damage != effect.value:
                raise ValueError(
                    f"All {move.__class__.__name__}'s damage instances must be of the same value"
                )

            intent.instances += 1

        if not intent.block and effect.type == EffectType.GAIN_BLOCK:
            intent.block = True

        # TODO: add support for other buffs
        if not intent.buff and effect.type == EffectType.GAIN_STR:
            intent.buff = True

    return intent


def _monster_to_view(entities: Entities, monster_entity_id: int) -> MonsterView:
    monster = entities.get_entity(monster_entity_id)
    actor_view = _actor_to_view(monster)
    intent_view = _move_to_intent(monster.move)

    return MonsterView(
        actor_view.name,
        actor_view.health,
        actor_view.block,
        actor_view.modifiers,
        monster_entity_id,  # TODO: revisit order
        intent_view,
    )


def view_monsters(entities: Entities) -> list[MonsterView]:
    return [
        _monster_to_view(entities, monster_entity_id) for monster_entity_id in entities.monster_ids
    ]
