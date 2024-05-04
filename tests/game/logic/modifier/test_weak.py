from src.agents.random import RandomAgent
from src.game.battle.engine import BattleEngine
from src.game.context import Context
from src.game.context import EntityData
from src.game.logic.card.strike import DAMAGE
from src.game.pipeline.steps.apply_weak import \
    WEAK_FACTOR  # TODO: create game constants module?


MONSTER_ENTITY_ID = 1


# TODO: test that stacks decrease by 1 @ end of turn
def test_base():
    # Instantiate the Agent, Context, and Engine
    agent = RandomAgent()
    context = Context(
        entities={
            Context.CHAR_ENTITY_ID: EntityData(name="Silent", max_health=50),
            MONSTER_ENTITY_ID: EntityData(name="Dummy", max_health=50),
        },
        entity_modifiers={(Context.CHAR_ENTITY_ID, "Weak"): 1},
        hand=["Strike"],
        active_card_idx=0,
    )
    engine = BattleEngine(agent, context)

    # Store previous health
    prev_health = context.entities[MONSTER_ENTITY_ID].current_health

    # Play the card
    engine._play_card(MONSTER_ENTITY_ID)

    # Assert that the monster's health has decreased by int(DAMAGE * WEAK_FACTOR)
    assert context.entities[MONSTER_ENTITY_ID].current_health == prev_health - int(
        DAMAGE * WEAK_FACTOR
    )
