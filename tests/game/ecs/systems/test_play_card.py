import pytest

from src.game.ecs.components.cards import CardCostComponent
from src.game.ecs.components.cards import CardHasEffectsComponent
from src.game.ecs.components.cards import CardIsActiveComponent
from src.game.ecs.components.effects import EffectSelectionType
from src.game.ecs.components.effects import EffectSelectionTypeComponent
from src.game.ecs.components.effects import EffectToBeDispatchedComponent
from src.game.ecs.components.energy import EnergyComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.play_card import PlayCardSystem


def test_base() -> None:
    # Instance ECS manager
    manager = ECSManager()

    # Create `num_effects` fake effect entities
    num_effects = 3
    effect_entity_ids = [
        manager.create_entity(EffectSelectionTypeComponent(EffectSelectionType.NONE))
        for _ in range(num_effects)
    ]

    # Create a card with those effects
    card_cost = 1
    manager.create_entity(
        CardHasEffectsComponent(effect_entity_ids),
        CardCostComponent(card_cost),
        CardIsActiveComponent(),
    )

    # Create an energy entity
    base_energy = 3
    energy_entity_id = manager.create_entity(EnergyComponent(base_energy))

    # Run the system
    PlayCardSystem().process(manager)

    # Verify there effects where tagged to be dispatched in the correct order
    for priority, effect_entity_id in enumerate(effect_entity_ids):
        assert (
            priority
            == manager.get_component_for_entity(
                effect_entity_id, EffectToBeDispatchedComponent
            ).priority
        )

    # Verify the current energy has been decreased by `card_cost`
    assert (
        manager.get_component_for_entity(energy_entity_id, EnergyComponent).current
        == base_energy - card_cost
    )

    # Verifiy the card is no longer active
    assert len(list(manager.get_component(CardIsActiveComponent))) == 0


def test_insufficient_energy() -> None:
    # Instance ECS manager
    manager = ECSManager()

    # Create a card with `card_cost` cost
    card_cost = 1
    manager.create_entity(
        CardCostComponent(card_cost),
        CardIsActiveComponent(),
    )

    # Create an energy entity such that `base_energy` < `card_cost`
    base_energy = 0
    manager.create_entity(EnergyComponent(base_energy))

    # Run the system
    with pytest.raises(ValueError) as err_info:
        PlayCardSystem().process(manager)

    assert str(err_info.value) == "Not enough energy to play card"