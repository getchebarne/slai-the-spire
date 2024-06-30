from dataclasses import dataclass
from typing import Optional

from src.game.combat.context import GameContext
from src.game.combat.view.actor import ActorView
from src.game.combat.view.actor import _actor_to_view


@dataclass
class IntentView:
    damage: Optional[int]
    times: Optional[int]
    block: bool


@dataclass
class MonsterView(ActorView):
    pass
    # intent: IntentView


def view_monsters(context: GameContext) -> list[MonsterView]:
    return [_actor_to_view(monster) for monster in context.monsters]
