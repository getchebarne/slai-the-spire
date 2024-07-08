from src.game.combat.entities import Entities
from src.game.combat.factories import defend
from src.game.combat.factories import dummy
from src.game.combat.factories import energy
from src.game.combat.factories import neutralize
from src.game.combat.factories import silent
from src.game.combat.factories import strike
from src.game.combat.factories import survivor


def create_combat() -> Entities:
    # Create Entities instance
    entities = Entities()

    # Fill
    # TODO: create functions for this
    entities.character_id = entities.create_entity(silent())
    entities.monster_ids = [entities.create_entity(dummy())]
    entities.energy_id = entities.create_entity(energy())
    entities.card_in_deck_ids = {
        entities.create_entity(strike()),
        entities.create_entity(strike()),
        entities.create_entity(strike()),
        entities.create_entity(strike()),
        entities.create_entity(strike()),
        entities.create_entity(defend()),
        entities.create_entity(defend()),
        entities.create_entity(defend()),
        entities.create_entity(defend()),
        entities.create_entity(defend()),
        entities.create_entity(neutralize()),
        entities.create_entity(survivor()),
    }

    return entities
