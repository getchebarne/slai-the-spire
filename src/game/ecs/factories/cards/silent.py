from src.game.ecs.components.cards import CardCostComponent
from src.game.ecs.components.cards import CardInDeckComponent
from src.game.ecs.components.cards import CardInHandComponent
from src.game.ecs.components.common import DescriptionComponent
from src.game.ecs.components.common import NameComponent
from src.game.ecs.components.creatures import CharacterComponent
from src.game.ecs.components.creatures import MonsterComponent
from src.game.ecs.components.effects import DealDamageEffectComponent
from src.game.ecs.components.effects import DiscardEffectComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.components.effects import EffectSelectionType
from src.game.ecs.components.effects import EffectSelectionTypeComponent
from src.game.ecs.components.effects import GainBlockEffectComponent
from src.game.ecs.components.effects import GainWeakEffectComponent
from src.game.ecs.components.effects import HasEffectsComponent
from src.game.ecs.manager import ECSManager


def create_neutralize(manager: ECSManager) -> int:
    base_cost = 0
    base_damage = 3
    base_weak = 1

    # Create a "DealDamage" effect
    deal_damage_entity_id = manager.create_entity(
        DealDamageEffectComponent(base_damage),
        EffectQueryComponentsComponent([MonsterComponent]),
        EffectSelectionTypeComponent(EffectSelectionType.SPECIFIC),
    )
    # Create a "GainWeak" effect
    gain_weak_entity_id = manager.create_entity(
        GainWeakEffectComponent(base_weak),
        EffectQueryComponentsComponent([MonsterComponent]),
        EffectSelectionTypeComponent(EffectSelectionType.SPECIFIC),
    )
    # Create "Neutralize" card in the deck and return its id
    return manager.create_entity(
        CardInDeckComponent(),
        NameComponent("Neutralize"),
        DescriptionComponent("Deal 3 damage. Apply 1 Weak."),
        CardCostComponent(base_cost),
        HasEffectsComponent([deal_damage_entity_id, gain_weak_entity_id]),
    )


def create_survivor(manager: ECSManager) -> int:
    base_cost = 1
    base_block = 8
    base_discard = 1

    # Create a "GainBlock" effect
    gain_block_entity_id = manager.create_entity(
        GainBlockEffectComponent(base_block),
        EffectQueryComponentsComponent([CharacterComponent]),
        EffectSelectionTypeComponent(EffectSelectionType.NONE),
    )
    # Create a "Discard" effect
    discard_entity_id = manager.create_entity(
        DiscardEffectComponent(base_discard),
        EffectQueryComponentsComponent([CardInHandComponent]),
        EffectSelectionTypeComponent(EffectSelectionType.SPECIFIC),
    )
    # Create "Survivor" card in the deck and return its id
    return manager.create_entity(
        CardInDeckComponent(),
        NameComponent("Survivor"),
        DescriptionComponent("Gain 8 block. Discard 1 card."),
        CardCostComponent(base_cost),
        HasEffectsComponent([gain_block_entity_id, discard_entity_id]),
    )
