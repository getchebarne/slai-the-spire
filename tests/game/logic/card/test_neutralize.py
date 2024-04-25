from src.agents.random import RandomAgent
from src.game.battle.engine import BattleEngine
from src.game.context import Context
from src.game.context import EntityData
from src.game.logic.card.neutralize import DAMAGE
from src.game.logic.card.neutralize import WEAK


MONSTER_ENTITY_ID = 1


def test_base():
    # Instantiate the Agent, Context, and Engine
    agent = RandomAgent()
    context = Context(
        entities={
            Context.CHAR_ENTITY_ID: EntityData(name="Silent", max_health=50),
            MONSTER_ENTITY_ID: EntityData(name="Dummy", max_health=50),
        },
        hand=["Neutralize"],
        active_card_idx=0,
    )
    engine = BattleEngine(agent, context)

    # Store previous health & weak
    prev_health = context.entities[MONSTER_ENTITY_ID].current_health
    prev_weak = context.entity_modifiers[(MONSTER_ENTITY_ID, "Weak")]

    # Play the card
    engine._play_card(MONSTER_ENTITY_ID)

    # Assert that the monster's health has decreased by DAMAGE and that the monster's "Weak"
    # modifier has increased by WEAK
    assert context.entities[MONSTER_ENTITY_ID].current_health == prev_health - DAMAGE
    assert context.entity_modifiers[(MONSTER_ENTITY_ID, "Weak")] == prev_weak + WEAK
