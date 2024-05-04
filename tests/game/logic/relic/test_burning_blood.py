from src.agents.random import RandomAgent
from src.game.battle.engine import BattleEngine
from src.game.context import Context
from src.game.context import EntityData
from src.game.logic.relic.burning_blood import HEAL


MONSTER_ENTITY_ID = 1


def test_base():
    # Instantiate the Agent, Context, and Engine
    agent = RandomAgent()
    context = Context(
        entities={
            Context.CHAR_ENTITY_ID: EntityData(name="Silent", max_health=50, current_health=25),
            MONSTER_ENTITY_ID: EntityData(name="Dummy", max_health=50),
        },
        relics=["Burning Blood"],
    )
    engine = BattleEngine(agent, context)

    # Store previous health
    prev_health = context.entities[Context.CHAR_ENTITY_ID].current_health

    # Assert that the character's health remains unchanged after the character's start of turn,
    # end of turn, and start of battle
    engine.battle_start()
    engine._char_turn_start()
    engine._char_turn_end()
    assert context.entities[Context.CHAR_ENTITY_ID].current_health == prev_health

    # End the battle
    engine.battle_end()

    # Assert that the character's health is increased by HEAL
    assert context.entities[Context.CHAR_ENTITY_ID].current_health == prev_health + HEAL
