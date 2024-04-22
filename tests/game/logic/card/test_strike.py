from src.game.context import Context
from src.game.context import EntityData
from src.game.battle.engine import BattleEngine
from src.agents.random import RandomAgent
from src.game.logic.card.strike import DAMAGE


def test_base():
    # Instantiate the Agent, Context, and Engine
    agent = RandomAgent()
    context = Context(
        entities={
            0: EntityData(name="Silent", max_health=50),
            1: EntityData(name="Dummy", max_health=50),
        },
        hand=["Strike"],
        active_card_idx=0,
    )
    engine = BattleEngine(agent, context)

    # Store previous health
    prev_block = context.get_char()[1].current_health

    # Play the card
    engine._play_card(monster_entity_id=1)

    # Assert that the character's health has decreased by DAMAGE
    assert context.entities[1].current_health == prev_block - DAMAGE
