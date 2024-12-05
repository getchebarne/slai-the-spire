from src.game.combat.effect_queue import EffectQueue
from src.game.combat.entities import Entities
from src.game.combat.factories import defend
from src.game.combat.factories import energy
from src.game.combat.factories import jaw_worm
from src.game.combat.factories import neutralize
from src.game.combat.factories import silent
from src.game.combat.factories import strike
from src.game.combat.factories import survivor
from src.game.combat.manager import CombatManager


def create_combat_manager() -> CombatManager:
    # Create entities
    # TODO: create functions for this
    entities = Entities()
    entities.character_id = entities.create_entity(silent())
    entities.monster_ids = [entities.create_entity(jaw_worm())]
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
    # Create effect queue
    effect_queue = EffectQueue()

    # Create combat manager w/ entities and effect queue
    combat_manager = CombatManager(entities=entities, effect_queue=effect_queue)

    return combat_manager
