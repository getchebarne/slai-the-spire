from dataclasses import dataclass
from typing import Optional

from src.game.combat.state import EffectType
from src.game.combat.state import GameState
from src.game.combat.state import MonsterMove
from src.game.combat.utils import card_requires_target
from src.game.combat.view.actor import ActorView
from src.game.combat.view.actor import _actor_to_view


@dataclass
class IntentView:
    damage: Optional[tuple[int, int]]
    block: bool


@dataclass
class MonsterView(ActorView):
    entity_id: int
    intent: IntentView
    is_selectable: bool


def _move_to_intent(move: MonsterMove) -> IntentView:
    # Initialze empty intent
    intent = IntentView(None, False)

    # Iterate over the move's effects
    for effect in move.effects:
        if effect.type == EffectType.DEAL_DAMAGE:
            if intent.damage is None:
                intent.damage = (effect.value, 1)

                continue

            if intent.damage[0] != effect.value:
                raise ValueError(
                    f"All {move.__class__.__name__}'s damage instances must be of the same value"
                )

            intent.damage[1] += 1

        if not intent.block and effect.type == EffectType.GAIN_BLOCK:
            intent.block = True

    return intent


def _monster_to_view(state: GameState, monster_entity_id: int) -> MonsterView:
    monster = state.get_entity(monster_entity_id)
    actor_view = _actor_to_view(monster)
    intent_view = _move_to_intent(monster.move)

    is_selectable = False
    if state.card_active_id is not None and card_requires_target(
        state.get_entity(state.card_active_id)
    ):
        is_selectable = True

    return MonsterView(
        actor_view.name,
        actor_view.health,
        actor_view.block,
        actor_view.modifiers,
        monster_entity_id,  # TODO: revisit order
        intent_view,
        is_selectable,
    )


def view_monsters(state: GameState) -> list[MonsterView]:
    return [_monster_to_view(state, monster_entity_id) for monster_entity_id in state.monster_ids]
