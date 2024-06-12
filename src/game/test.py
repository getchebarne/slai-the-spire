from src.agents.random import RandomAgent
from src.game.combat.engine import CombatEngine
from src.game.ecs.factories.creatures.characters import create_silent
from src.game.ecs.factories.creatures.monsters import create_dummy
from src.game.ecs.factories.decks.silent import create_starter_deck
from src.game.ecs.factories.energy import create_energy
from src.game.ecs.manager import ECSManager


agent = RandomAgent()
combat_engine = CombatEngine()
manager = ECSManager()

create_starter_deck(manager)
create_energy(manager)

dummy_entity_id = create_dummy(manager)
silent_entity_id = create_silent(manager)

combat_engine.run(manager, agent)
