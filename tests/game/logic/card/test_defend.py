from src.game.context import Context
from src.game.context import EntityData
from src.game.battle.engine import BattleEngine
from src.agents.random import RandomAgent
from src.game.logic.card.defend import BLOCK_AMOUNT


def test_base():
    # Instantiate the Agent, Context, and Engine
    agent = RandomAgent()
    context = Context(
        entities={
            0: EntityData(name="Silent", max_health=50),
            1: EntityData(name="Dummy", max_health=50),
        },
        hand=["Defend"],
        active_card_idx=0,
    )
    engine = BattleEngine(agent, context)

    # Store previous block
    prev_block = context.get_char()[1].current_block

    # Play the card
    engine._play_card()

    # Assert that the character's block has increased by BLOCK_AMOUNT
    assert context.get_char()[1].current_block == prev_block + BLOCK_AMOUNT
