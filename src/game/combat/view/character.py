from dataclasses import dataclass

from src.game.combat.context import GameContext
from src.game.combat.view.actor import ActorView
from src.game.combat.view.actor import _actor_to_view


@dataclass
class CharacterView(ActorView):
    pass


def view_character(context: GameContext) -> CharacterView:
    actor_view = _actor_to_view(context.character)

    return CharacterView(actor_view.name, actor_view.health, actor_view.block)
