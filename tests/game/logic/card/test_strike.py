from src.agents.random import RandomAgent
from src.game.battle.engine import BattleEngine
from src.game.context import Context
from src.game.context import EntityData
from src.game.logic.card.strike import DAMAGE


MONSTER_ENTITY_ID = 1


def test_base():
    # Instantiate the Agent, Context, and Engine
    agent = RandomAgent()
    context = Context(
        entities={
            0: EntityData(name="Silent", max_health=50),
            MONSTER_ENTITY_ID: EntityData(name="Dummy", max_health=50),
        },
        hand=["Strike"],
        active_card_idx=0,
    )
    engine = BattleEngine(agent, context)

    # Store previous health
    prev_block = context.entities[context.CHAR_ENTITY_ID].current_health

    # Play the card
    engine._play_card(MONSTER_ENTITY_ID)

    # Assert that the character's health has decreased by DAMAGE
    assert context.entities[MONSTER_ENTITY_ID].current_health == prev_block - DAMAGE
