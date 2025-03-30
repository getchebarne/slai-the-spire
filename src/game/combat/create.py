from src.game.combat.entities import EntityManager
from src.game.combat.entities import create_entity
from src.game.combat.factories import create_backflip
from src.game.combat.factories import create_dagger_throw
from src.game.combat.factories import create_dash
from src.game.combat.factories import create_defend
from src.game.combat.factories import create_energy
from src.game.combat.factories import create_jaw_worm
from src.game.combat.factories import create_leg_sweep
from src.game.combat.factories import create_neutralize
from src.game.combat.factories import create_silent
from src.game.combat.factories import create_strike
from src.game.combat.factories import create_survivor
from src.game.combat.state import CombatState


# TODO: parametrize deck, monster, etc.
def create_combat_state() -> CombatState:
    # Create empty EntityManager
    entity_manager = EntityManager([])

    # Create entities
    id_charater = create_entity(entity_manager, create_silent(50, 50))
    id_monsters = [create_entity(entity_manager, create_jaw_worm())]
    id_energy = create_entity(entity_manager, create_energy(3, 3))
    id_cards_in_deck = [
        create_entity(entity_manager, create_strike()),
        create_entity(entity_manager, create_strike()),
        create_entity(entity_manager, create_strike()),
        create_entity(entity_manager, create_strike()),
        create_entity(entity_manager, create_strike()),
        create_entity(entity_manager, create_defend()),
        create_entity(entity_manager, create_defend()),
        create_entity(entity_manager, create_defend()),
        create_entity(entity_manager, create_defend()),
        create_entity(entity_manager, create_defend()),
        create_entity(entity_manager, create_survivor()),
        create_entity(entity_manager, create_neutralize()),
        create_entity(entity_manager, create_dash()),
        create_entity(entity_manager, create_backflip()),
        create_entity(entity_manager, create_dagger_throw()),
        create_entity(entity_manager, create_leg_sweep()),
    ]

    # Assign corresponding ids
    entity_manager.id_character = id_charater
    entity_manager.id_monsters = id_monsters
    entity_manager.id_energy = id_energy
    entity_manager.id_cards_in_deck = id_cards_in_deck

    return CombatState(entity_manager, [])
