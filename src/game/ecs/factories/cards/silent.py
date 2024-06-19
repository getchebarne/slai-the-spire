from src.game.ecs.components.cards import CardCostComponent
from src.game.ecs.components.cards import CardInDeckComponent
from src.game.ecs.components.cards import CardNeutralizeComponent
from src.game.ecs.components.cards import CardRequiresTargetComponent
from src.game.ecs.components.cards import CardSurvivorComponent
from src.game.ecs.components.common import DescriptionComponent
from src.game.ecs.components.common import NameComponent
from src.game.ecs.manager import ECSManager


def create_neutralize(manager: ECSManager) -> int:
    base_cost = 0

    # Create "Neutralize" card in deck and return its id
    return manager.create_entity(
        NameComponent("Neutralize"),
        DescriptionComponent("Deal 3 damage. Apply 1 Weak."),
        CardNeutralizeComponent(),
        CardInDeckComponent(),
        CardCostComponent(base_cost),
        CardRequiresTargetComponent(),
    )


def create_survivor(manager: ECSManager) -> int:
    base_cost = 1

    # Create "Survivor" card in deck and return its id
    return manager.create_entity(
        NameComponent("Survivor"),
        DescriptionComponent("Gain 8 block. Discard 1 card."),
        CardSurvivorComponent(),
        CardInDeckComponent(),
        CardCostComponent(base_cost),
    )
