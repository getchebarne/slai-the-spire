from src.game.ecs.components.cards import CardCostComponent
from src.game.ecs.components.cards import CardDefendComponent
from src.game.ecs.components.cards import CardInDeckComponent
from src.game.ecs.components.cards import CardRequiresTargetComponent
from src.game.ecs.components.cards import CardStrikeComponent
from src.game.ecs.components.common import DescriptionComponent
from src.game.ecs.components.common import NameComponent
from src.game.ecs.manager import ECSManager


def create_strike(manager: ECSManager) -> int:
    base_cost = 1

    # Create "Strike" card in deck and return its entity_id
    return manager.create_entity(
        NameComponent("Strike"),
        DescriptionComponent("Deal 6 damage."),
        CardStrikeComponent(),
        CardInDeckComponent(),
        CardCostComponent(base_cost),
        CardRequiresTargetComponent(),
    )


def create_defend(manager: ECSManager) -> int:
    base_cost = 1

    # Create "Defend" card in deck and return its id
    return manager.create_entity(
        NameComponent("Defend"),
        DescriptionComponent("Gain 5 block."),
        CardDefendComponent(),
        CardInDeckComponent(),
        CardCostComponent(base_cost),
    )
