from src.agents.random import RandomAgent
from src.game.battle.engine import BattleEngine
from src.game.context import Context
from src.game.context import EntityData


MONSTER_ENTITY_ID = 1
POISON_STACKS = 2


def test_base():
    # Instantiate the Agent, Context, and Engine
    agent = RandomAgent()
    context = Context(
        entities={
            Context.CHAR_ENTITY_ID: EntityData(name="Silent", max_health=50),
            MONSTER_ENTITY_ID: EntityData(name="Dummy", max_health=50),
        },
        entity_modifiers={(MONSTER_ENTITY_ID, "Poison"): POISON_STACKS},
    )
    engine = BattleEngine(agent, context)

    # Store previous health
    prev_health = context.entities[MONSTER_ENTITY_ID].current_health

    # Call the monster's start of turn
    engine._monsters_turn_start()

    # Assert that the monster's health has decreased by POISON_STACKS
    assert context.entities[MONSTER_ENTITY_ID].current_health == prev_health - POISON_STACKS

    # Call the monster's end of turn
    engine._monsters_turn_end()

    # Assert that the monster's poison stacks have decreased by 1
    assert context.entity_modifiers[(MONSTER_ENTITY_ID, "Poison")] == POISON_STACKS - 1

    # TODO: test that poison pierces block
