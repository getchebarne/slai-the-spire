from src.agents.random import RandomAgent
from src.game.battle.engine import BattleEngine
from src.game.context import Context
from src.game.context import EntityData
from src.game.logic.card.strike import DAMAGE


MONSTER_ENTITY_ID = 1
STRENGTHS = [1, 2, 3, 4, 5]


def test_base():
    # Instantiate the Agent, Context, and Engine
    agent = RandomAgent()
    context = Context(
        entities={
            Context.CHAR_ENTITY_ID: EntityData(name="Silent", max_health=50),
            MONSTER_ENTITY_ID: EntityData(name="Dummy", max_health=50),
        }
    )
    engine = BattleEngine(agent, context)

    for strength in STRENGTHS:
        # Reset the monster's health
        context.entities[MONSTER_ENTITY_ID].current_health = context.entities[
            MONSTER_ENTITY_ID
        ].max_health

        # Set the character's strength
        context.entity_modifiers[(context.CHAR_ENTITY_ID, "Strength")] = strength

        # Set hand to a single "Strike" and make it active
        context.hand = ["Strike"]
        context.active_card_idx = 0

        # Store previous health
        prev_health = context.entities[MONSTER_ENTITY_ID].current_health

        # Play the card
        engine._play_card(MONSTER_ENTITY_ID)

        # Assert that the monster's health has decreased by DAMAGE + strength
        assert (
            context.entities[MONSTER_ENTITY_ID].current_health == prev_health - DAMAGE - strength
        )
