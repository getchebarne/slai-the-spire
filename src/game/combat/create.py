from collections import deque

from src.game.engine.state import GameState
from src.game.entity.manager import EntityManager
from src.game.entity.manager import create_entity
from src.game.factory.card.backflip import create_card_backflip
from src.game.factory.card.dagger_throw import create_card_dagger_throw
from src.game.factory.card.dash import create_card_dash
from src.game.factory.card.defend import create_card_defend
from src.game.factory.card.leg_sweep import create_card_leg_sweep
from src.game.factory.card.neutralize import create_card_neutralize
from src.game.factory.card.strike import create_card_strike
from src.game.factory.card.survivor import create_card_survivor
from src.game.factory.character.silent import create_character_silent
from src.game.factory.energy import create_energy
from src.game.factory.monster.cultist import create_monster_cultist
from src.game.factory.monster.fungi_beast import create_monster_fungi_beast
from src.game.factory.monster.jaw_worm import create_monster_jaw_worm


# TODO: parametrize deck, monster, etc.
def create_combat_state() -> GameState:
    # Create empty EntityManager
    entity_manager = EntityManager([])

    # Create entities
    id_charater = create_entity(entity_manager, create_character_silent(15, 15))
    id_monsters = [
        # create_entity(entity_manager, create_monster_cultist()),
        create_entity(entity_manager, create_monster_fungi_beast()),
        create_entity(entity_manager, create_monster_fungi_beast()),
    ]
    id_energy = create_entity(entity_manager, create_energy(3, 3))
    id_cards_in_deck = [
        create_entity(entity_manager, create_card_strike()),
        create_entity(entity_manager, create_card_strike()),
        create_entity(entity_manager, create_card_strike()),
        create_entity(entity_manager, create_card_strike()),
        create_entity(entity_manager, create_card_strike()),
        create_entity(entity_manager, create_card_defend()),
        create_entity(entity_manager, create_card_defend()),
        create_entity(entity_manager, create_card_defend()),
        create_entity(entity_manager, create_card_defend()),
        create_entity(entity_manager, create_card_defend()),
        create_entity(entity_manager, create_card_survivor()),
        create_entity(entity_manager, create_card_neutralize()),
        create_entity(entity_manager, create_card_dash()),
        create_entity(entity_manager, create_card_backflip()),
        create_entity(entity_manager, create_card_dagger_throw()),
        create_entity(entity_manager, create_card_leg_sweep()),
    ]

    # Assign corresponding ids
    entity_manager.id_character = id_charater
    entity_manager.id_monsters = id_monsters
    entity_manager.id_energy = id_energy
    entity_manager.id_cards_in_deck = id_cards_in_deck

    return GameState(entity_manager, deque(), None)
