from dataclasses import dataclass

from src.game.combat.state import GameState
from src.game.combat.view.actor import ActorView
from src.game.combat.view.actor import _actor_to_view


@dataclass
class CharacterView(ActorView):
    pass


def view_character(state: GameState) -> CharacterView:
    actor_view = _actor_to_view(state.get_character())

    return CharacterView(
        actor_view.name, actor_view.health, actor_view.block, actor_view.modifiers
    )
