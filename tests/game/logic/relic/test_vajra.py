from src.agents.random import RandomAgent
from src.game.battle.engine import BattleEngine
from src.game.context import Context
from src.game.context import EntityData
from src.game.logic.relic.vajra import PLUS_STR


MONSTER_ENTITY_ID = 1


def test_base():
    # Instantiate the Agent, Context, and Engine
    agent = RandomAgent()
    context = Context(
        entities={
            Context.CHAR_ENTITY_ID: EntityData(name="Silent", max_health=50),
            MONSTER_ENTITY_ID: EntityData(name="Dummy", max_health=50),
        },
        relics=["Vajra"],
    )
    engine = BattleEngine(agent, context)

    # Store previous strength
    prev_str = context.entity_modifiers[(context.CHAR_ENTITY_ID, "Strength")]

    # Start the battle
    engine.battle_start()

    # Assert that the character's strength has increased by PLUS_STR
    assert context.entity_modifiers[(context.CHAR_ENTITY_ID, "Strength")] == prev_str + PLUS_STR
