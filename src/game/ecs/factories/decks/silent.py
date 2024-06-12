from src.game.ecs.factories.cards.colorless import create_defend
from src.game.ecs.factories.cards.colorless import create_strike

# from src.game.ecs.factories.cards.silent import create_neutralize
from src.game.ecs.factories.cards.silent import create_survivor
from src.game.ecs.manager import ECSManager


def create_starter_deck(manager: ECSManager) -> list[int]:
    return [
        create_strike(manager),
        create_strike(manager),
        create_strike(manager),
        create_strike(manager),
        create_strike(manager),
        create_defend(manager),
        create_defend(manager),
        create_defend(manager),
        create_defend(manager),
        create_defend(manager),
        create_survivor(manager),
        # create_neutralize(manager),
    ]
