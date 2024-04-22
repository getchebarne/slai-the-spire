from src.agents.random import RandomAgent
from src.game.battle.engine import BattleEngine
from src.game.context import Context
from src.game.context import EntityData
from src.game.logic.card.defend import BLOCK


MONSTER_ENTITY_ID = 1


def test_base():
    # Instantiate the Agent, Context, and Engine
    agent = RandomAgent()
    context = Context(
        entities={
            0: EntityData(name="Silent", max_health=50),
            MONSTER_ENTITY_ID: EntityData(name="Dummy", max_health=50),
        },
        hand=["Defend"],
        active_card_idx=0,
    )
    engine = BattleEngine(agent, context)

    # Store previous block
    prev_block = context.entities[context.CHAR_ENTITY_ID].current_block

    # Play the card
    engine._play_card()

    # Assert that the character's block has increased by BLOCK
    assert context.entities[context.CHAR_ENTITY_ID].current_block == prev_block + BLOCK
