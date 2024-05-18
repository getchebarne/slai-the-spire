import random

from src.game.ecs.components.cards import ActiveCardComponent
from src.game.ecs.components.cards import CardCostComponent
from src.game.ecs.components.effects import EffectApplyToComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.components.effects import EffectSelectionType
from src.game.ecs.components.effects import EffectSelectionTypeComponent
from src.game.ecs.components.effects import EffectToBeTargetedComponent
from src.game.ecs.components.effects import HasEffectsComponent
from src.game.ecs.components.energy import EnergyComponent
from src.game.ecs.components.target import TargetComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


class PlayCardSystem(BaseSystem):
    def __call__(self, manager: ECSManager) -> None:
        # TODO: decrease energy
        # TODO: make sure there's only one active card
        card_entity_id, active_card_component = list(manager.get_component(ActiveCardComponent))[0]

        # Get the card's cost
        card_cost = manager.get_component_for_entity(card_entity_id, CardCostComponent).value

        # Check if there's enough energy to play the card
        energy_component = list(manager.get_component(EnergyComponent))[0][1]
        if energy_component.current < card_cost:
            raise ValueError("Not enough energy to play card")

        # Decrease the energy. TODO: should this be an effect?
        energy_component.current -= card_cost

        # Flag the card's effects to be targeted
        for effect_entity_id in manager.get_component_for_entity(
            card_entity_id, HasEffectsComponent
        ).entity_ids:
            manager.add_component(effect_entity_id, EffectToBeTargetedComponent())

        # Remove the active card
        manager.remove_component(card_entity_id, ActiveCardComponent)


# TODO: add effect source to trigger additional effects (e.g., thorns)
class TargetEffectsSystem(BaseSystem):
    def __call__(self, manager: ECSManager) -> None:
        # Create a list of effect entities to untag after targeting
        effect_entity_ids_to_untag = []

        # Iterate over the effects to be targeted
        for effect_entity_id, (
            _,
            effect_selection_type_component,
            effect_query_components_component,
        ) in manager.get_components(
            EffectToBeTargetedComponent,
            EffectSelectionTypeComponent,
            EffectQueryComponentsComponent,
        ):
            # Get the entities that match the effect's potential targets
            query_target_entites = [
                query_entity_id
                for query_entity_id, _ in manager.get_components(
                    *effect_query_components_component.value
                )
            ]
            # Resolve target entity
            if effect_selection_type_component.value == EffectSelectionType.NONE:
                if len(query_target_entites) > 1:
                    raise ValueError("Too many entities to apply effect")

                target_entity_ids = query_target_entites

            elif effect_selection_type_component.value == EffectSelectionType.SPECIFIC:
                # TODO: improve
                target_entity_id, _ = list(manager.get_component(TargetComponent))[0]
                if target_entity_id not in query_target_entites:
                    raise ValueError(
                        f"Target entity {target_entity_id} not in query target entities"
                    )

                target_entity_ids = [target_entity_id]

            elif effect_selection_type_component.value == EffectSelectionType.RANDOM:
                target_entity_ids = [random.choice(query_target_entites)]

            elif effect_selection_type_component.value == EffectSelectionType.ALL:
                target_entity_ids = query_target_entites

            # Add a component to the effect entity to specify the entities to apply the effect to
            manager.add_component(
                effect_entity_id, EffectApplyToComponent(entity_ids=target_entity_ids)
            )

        # Untag the effects
        for effect_entity_id in effect_entity_ids_to_untag:
            manager.remove_component(effect_entity_id, EffectToBeTargetedComponent)
