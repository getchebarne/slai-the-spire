from src.game.ecs.components.cards import CardCostComponent
from src.game.ecs.components.cards import CardHasEffectsComponent
from src.game.ecs.components.cards import CardInDeckComponent
from src.game.ecs.components.common import DescriptionComponent
from src.game.ecs.components.common import IsSelectedComponent
from src.game.ecs.components.common import NameComponent
from src.game.ecs.components.creatures import CharacterComponent
from src.game.ecs.components.creatures import MonsterComponent
from src.game.ecs.components.effects import EffectDealDamageComponent
from src.game.ecs.components.effects import EffectGainBlockComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.components.effects import EffectSelectionType
from src.game.ecs.components.effects import EffectSelectionTypeComponent
from src.game.ecs.manager import ECSManager


def create_strike(manager: ECSManager) -> int:
    base_cost = 1
    base_damage = 6

    # Create a "DealDamage" effect
    deal_damage_entity_id = manager.create_entity(
        EffectDealDamageComponent(base_damage),
        EffectQueryComponentsComponent([MonsterComponent, IsSelectedComponent]),
        EffectSelectionTypeComponent(EffectSelectionType.NONE),
    )

    # Create "Strike" card in deck and return its entity_id
    return manager.create_entity(
        CardInDeckComponent(),
        NameComponent("Strike"),
        DescriptionComponent("Deal 6 damage."),
        CardCostComponent(base_cost),
        CardHasEffectsComponent([deal_damage_entity_id]),
    )


def create_defend(manager: ECSManager) -> int:
    base_cost = 1
    base_block = 5

    # Create a "GainBlock" effect
    gain_block_entity_id = manager.create_entity(
        EffectGainBlockComponent(base_block),
        EffectQueryComponentsComponent([CharacterComponent]),
        EffectSelectionTypeComponent(EffectSelectionType.NONE),  # TODO: revisit
    )

    # Create "Defend" card in deck and return its id
    return manager.create_entity(
        CardInDeckComponent(),
        NameComponent("Defend"),
        DescriptionComponent("Gain 5 block."),
        CardCostComponent(base_cost),
        CardHasEffectsComponent([gain_block_entity_id]),
    )
