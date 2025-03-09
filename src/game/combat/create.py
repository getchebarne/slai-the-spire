from dataclasses import replace

from src.game.combat.entities import Entities
from src.game.combat.entities import add_entities
from src.game.combat.factories import create_backflip
from src.game.combat.factories import create_dagger_throw
from src.game.combat.factories import create_dash
from src.game.combat.factories import create_defend
from src.game.combat.factories import create_dummy
from src.game.combat.factories import create_energy
from src.game.combat.factories import create_leg_sweep
from src.game.combat.factories import create_neutralize
from src.game.combat.factories import create_silent
from src.game.combat.factories import create_strike
from src.game.combat.factories import create_survivor
from src.game.combat.state import CombatState


# TODO: improve double assign
def create_combat_state() -> CombatState:
    # Create entities
    entities = Entities()

    # Character
    entities, character_id = add_entities(entities, create_silent(50, 50))
    entities = replace(entities, character_id=character_id[0])

    # Monsters
    entities, monster_ids = add_entities(entities, create_dummy(50, 50, None))
    entities = replace(entities, monster_ids=monster_ids)

    # Energy
    entities, energy_id = add_entities(entities, create_energy())
    entities = replace(entities, energy_id=energy_id[0])

    # Deck
    entities, card_in_deck_ids = add_entities(
        entities,
        create_strike(),
        create_strike(),
        create_strike(),
        create_strike(),
        create_strike(),
        create_defend(),
        create_defend(),
        create_defend(),
        create_defend(),
        create_defend(),
        # create_neutralize(),
        # create_survivor(),
        # create_dagger_throw(),
        # create_leg_sweep(),
        # create_backflip(),
        # create_dash(),
    )
    entities = replace(entities, card_in_deck_ids=card_in_deck_ids)

    # Create effect queue
    effect_queue = []

    # Return combat state w/ entities and effect queue
    return CombatState(entities=entities, effect_queue=effect_queue)
