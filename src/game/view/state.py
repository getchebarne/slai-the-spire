from dataclasses import dataclass

from src.game.state import GameState
from src.game.view.card import ViewCard
from src.game.view.card import get_view_deck
from src.game.view.card import get_view_hand
from src.game.view.card import get_view_pile_disc
from src.game.view.card import get_view_pile_draw
from src.game.view.card import get_view_pile_exhaust
from src.game.view.card import get_view_reward_combat
from src.game.view.character import ViewCharacter
from src.game.view.character import get_view_character
from src.game.view.energy import ViewEnergy
from src.game.view.energy import get_view_energy
from src.game.view.fsm import ViewFSM
from src.game.view.map_ import ViewMap
from src.game.view.map_ import get_view_map
from src.game.view.monster import ViewMonster
from src.game.view.monster import get_view_monsters


@dataclass(frozen=True)
class ViewGameState:
    character: ViewCharacter
    monsters: list[ViewMonster]
    deck: list[ViewCard]
    hand: list[ViewCard]
    pile_draw: list[ViewCard]
    pile_disc: list[ViewCard]
    pile_exhaust: list[ViewCard]
    reward_combat: list[ViewCard]
    energy: ViewEnergy
    map: ViewMap
    fsm: ViewFSM


def get_view_game_state(game_state: GameState) -> ViewGameState:
    entity_manager = game_state.entity_manager

    return ViewGameState(
        get_view_character(entity_manager),
        get_view_monsters(entity_manager),
        get_view_deck(entity_manager),
        get_view_hand(entity_manager),
        get_view_pile_draw(entity_manager),
        get_view_pile_disc(entity_manager),
        get_view_pile_exhaust(entity_manager),
        get_view_reward_combat(entity_manager),
        get_view_energy(entity_manager),
        get_view_map(entity_manager),
        game_state.fsm,
    )
