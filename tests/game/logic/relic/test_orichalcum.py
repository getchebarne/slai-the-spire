from src.agents.random import RandomAgent
from src.game.battle.engine import BattleEngine
from src.game.context import Context
from src.game.context import EntityData
from src.game.logic.relic.orichalcum import BLOCK


MONSTER_ENTITY_ID = 1


def test_base():
    # Instantiate the Agent, Context, and Engine
    agent = RandomAgent()
    context = Context(
        entities={
            Context.CHAR_ENTITY_ID: EntityData(name="Silent", max_health=50, current_block=0),
            MONSTER_ENTITY_ID: EntityData(name="Dummy", max_health=50),
        },
        relics=["Orichalcum"],
    )
    engine = BattleEngine(agent, context)

    # Store previous block
    prev_block = context.entities[Context.CHAR_ENTITY_ID].current_block

    # Assert that the character's block remains unchanged after the character's start of turn,
    # start of battle, and end of battle
    engine._char_turn_start()
    engine.battle_start()
    engine.battle_end()
    assert context.entities[Context.CHAR_ENTITY_ID].current_block == prev_block

    # End the battle
    engine._char_turn_end()

    # Assert that the character's block is increased by BLOCK
    assert context.entities[Context.CHAR_ENTITY_ID].current_block == prev_block + BLOCK

    # Make sure this only works if the character's block is 0
    engine._char_turn_end()
    assert context.entities[Context.CHAR_ENTITY_ID].current_block == prev_block + BLOCK
