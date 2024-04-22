from src.agents.random import RandomAgent
from src.game.battle.engine import BattleEngine
from src.game.context import Context
from src.game.context import EntityData
from src.game.logic.card.neutralize import DAMAGE
from src.game.logic.card.neutralize import WEAK


def test_base():
    # Instantiate the Agent, Context, and Engine
    agent = RandomAgent()
    context = Context(
        entities={
            0: EntityData(name="Silent", max_health=50),
            1: EntityData(name="Dummy", max_health=50),
        },
        hand=["Neutralize"],
        active_card_idx=0,
    )
    engine = BattleEngine(agent, context)

    # Store previous health & weak
    prev_health = context.entities[context.CHAR_ENTITY_ID].current_health
    prev_weak = context.entity_modifiers[(1, "Weak")]

    # Play the card
    engine._play_card(monster_entity_id=1)

    # Assert that the monster's health has decreased by DAMAGE and that the monster's "Weak"
    # modifier has increased by WEAK
    assert context.entities[1].current_health == prev_health - DAMAGE
    assert context.entity_modifiers[(1, "Weak")] == prev_weak + WEAK
