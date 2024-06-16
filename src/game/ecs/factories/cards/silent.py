from src.game.ecs.components.cards import CardCostComponent
from src.game.ecs.components.cards import CardHasEffectsComponent
from src.game.ecs.components.cards import CardInDeckComponent
from src.game.ecs.components.cards import CardInHandComponent
from src.game.ecs.components.cards import CardTargetComponent
from src.game.ecs.components.common import DescriptionComponent
from src.game.ecs.components.common import NameComponent
from src.game.ecs.components.creatures import CharacterComponent
from src.game.ecs.components.creatures import ModifierWeakComponent
from src.game.ecs.components.creatures import MonsterComponent
from src.game.ecs.components.effects import EffectCreateWeakComponent
from src.game.ecs.components.effects import EffectDealDamageComponent
from src.game.ecs.components.effects import EffectDiscardCardComponent
from src.game.ecs.components.effects import EffectGainBlockComponent
from src.game.ecs.components.effects import EffectModifierDeltaComponent
from src.game.ecs.components.effects import EffectNumberOfTargetsComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.components.effects import EffectSelectionType
from src.game.ecs.components.effects import EffectSelectionTypeComponent
from src.game.ecs.manager import ECSManager


def create_neutralize(manager: ECSManager) -> int:
    base_cost = 0
    base_damage = 3
    base_weak = 1

    # Create a "DealDamage" effect
    deal_damage_entity_id = manager.create_entity(
        EffectDealDamageComponent(base_damage),
        EffectQueryComponentsComponent([MonsterComponent, CardTargetComponent]),
    )

    # Create a "GainWeak" effect
    create_weak_entity_id = manager.create_entity(
        EffectCreateWeakComponent(),
        EffectQueryComponentsComponent([MonsterComponent, CardTargetComponent]),
    )
    gain_weak_entity_id = manager.create_entity(
        EffectModifierDeltaComponent(base_weak),
        EffectQueryComponentsComponent(
            [ModifierWeakComponent, CardTargetComponent]
        ),  # TODO: revisit CardTarget here
    )
    # Create "Neutralize" card in deck and return its id
    return manager.create_entity(
        NameComponent("Neutralize"),
        DescriptionComponent("Deal 3 damage. Apply 1 Weak."),
        CardInDeckComponent(),
        CardCostComponent(base_cost),
        CardHasEffectsComponent(
            [deal_damage_entity_id, create_weak_entity_id, gain_weak_entity_id]
        ),
    )


def create_survivor(manager: ECSManager) -> int:
    base_cost = 1
    base_block = 8
    base_discard = 1

    # Create a "GainBlock" effect
    gain_block_entity_id = manager.create_entity(
        EffectGainBlockComponent(base_block), EffectQueryComponentsComponent([CharacterComponent])
    )

    # Create a "Discard" effect
    discard_entity_id = manager.create_entity(
        NameComponent("Discard"),
        EffectDiscardCardComponent(),
        EffectQueryComponentsComponent([CardInHandComponent]),
        EffectSelectionTypeComponent(EffectSelectionType.SPECIFIC),
        EffectNumberOfTargetsComponent(base_discard),
    )
    # Create "Survivor" card in deck and return its id
    return manager.create_entity(
        NameComponent("Survivor"),
        DescriptionComponent("Gain 8 block. Discard 1 card."),
        CardInDeckComponent(),
        CardCostComponent(base_cost),
        CardHasEffectsComponent([gain_block_entity_id, discard_entity_id]),
    )
