import random

from src.agents.random import RandomAgent
from src.game.battle.engine import BattleEngine
from src.game.context import Context
from src.game.context import EntityData
from src.game.logic.relic.mercury_hourglass import DAMAGE


HEALTH_LOWER = 40
HEALTH_UPPER = 50


def test_base():
    # Instantiate the Agent, Context, and Engine
    agent = RandomAgent()
    context = Context(
        entities={
            Context.CHAR_ENTITY_ID: EntityData(name="Silent", max_health=50),
            1: EntityData(name="Dummy", max_health=random.randint(HEALTH_LOWER, HEALTH_UPPER)),
            2: EntityData(name="Dummy", max_health=random.randint(HEALTH_LOWER, HEALTH_UPPER)),
            3: EntityData(name="Dummy", max_health=random.randint(HEALTH_LOWER, HEALTH_UPPER)),
        },
        relics=["Mercury Hourglass"],
    )
    engine = BattleEngine(agent, context)

    # Store previous health for the character and for each monster
    prev_char_health = context.entities[Context.CHAR_ENTITY_ID].current_health
    prev_monster_health = {
        entity_id: entitiy_data.current_health
        for entity_id, entitiy_data in context.get_monsters()
    }
    # Start character's turn
    engine._char_turn_start()

    # Assert that the character's health is unchanged
    assert context.entities[Context.CHAR_ENTITY_ID].current_health == prev_char_health

    # Assert that each monster's health is decreased by DAMAGE
    assert all(
        context.entities[entity_id].current_health == prev_health - DAMAGE
        for entity_id, prev_health in prev_monster_health.items()
    )
