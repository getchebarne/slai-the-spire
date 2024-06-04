from src.game.ecs.components.cards import CardCostComponent
from src.game.ecs.components.cards import CardHasEffectsComponent
from src.game.ecs.components.cards import CardInDeckComponent
from src.game.ecs.components.cards import CardRequiresTargetComponent
from src.game.ecs.components.common import DescriptionComponent
from src.game.ecs.components.common import NameComponent
from src.game.ecs.components.creatures import CharacterComponent
from src.game.ecs.components.creatures import MonsterComponent
from src.game.ecs.components.effects import DealDamageEffectComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.components.effects import EffectSelectionType
from src.game.ecs.components.effects import EffectSelectionTypeComponent
from src.game.ecs.components.effects import GainBlockEffectComponent
from src.game.ecs.manager import ECSManager


def create_strike(manager: ECSManager) -> int:
    base_cost = 1
    base_damage = 6

    # Create a "DealDamage" effect
    deal_damage_entity_id = manager.create_entity(
        DealDamageEffectComponent(base_damage),
        EffectQueryComponentsComponent([MonsterComponent]),
        EffectSelectionTypeComponent(EffectSelectionType.SPECIFIC),
    )

    # Create "Strike" card in the deck and return its entity_id
    return manager.create_entity(
        CardInDeckComponent(),
        NameComponent("Strike"),
        DescriptionComponent("Deal 6 damage."),
        CardCostComponent(base_cost),
        CardHasEffectsComponent([deal_damage_entity_id]),
        CardRequiresTargetComponent(),  # TODO: revisit, this might be not needed
    )


def create_defend(manager: ECSManager) -> int:
    base_cost = 1
    base_block = 5

    # Create a "GainBlock" effect
    gain_block_entity_id = manager.create_entity(
        GainBlockEffectComponent(base_block),
        EffectQueryComponentsComponent([CharacterComponent]),
        EffectSelectionTypeComponent(EffectSelectionType.NONE),  # TODO: revisit
    )

    # Create "Defend" card in the deck and return its id
    return manager.create_entity(
        CardInDeckComponent(),
        NameComponent("Defend"),
        DescriptionComponent("Gain 5 block."),
        CardCostComponent(base_cost),
        CardHasEffectsComponent([gain_block_entity_id]),
    )
