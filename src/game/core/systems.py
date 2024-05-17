import random
from abc import ABC, abstractmethod
from collections import defaultdict

from src.game.core.components import ActiveCardComponent
from src.game.core.components import CardInHandComponent
from src.game.core.components import EffectsOnUseComponent
from src.game.core.components import EffectsToBeAppliedComponent
from src.game.core.components import EffectsToBeTargetedComponent
from src.game.core.components import TargetComponent
from src.game.core.effect import Effect
from src.game.core.effect import SelectionType
from src.game.core.manager import ECSManager
from src.game.pipeline.pipeline import EffectPipeline


_effect_pipeline = EffectPipeline()


class BaseSystem(ABC):
    @abstractmethod
    def __call__(self, manager: ECSManager) -> None:
        pass


class PlayCardSystem(BaseSystem):
    def __call__(self, manager: ECSManager) -> None:
        # TODO: decrease energy
        # TODO: make sure there's only one active card
        entitiy_id, active_card_component = list(
            manager.get_components(ActiveCardComponent, CardInHandComponent)
        )[0]

        # Get the card's effects
        effects = manager.get_component_for_entity(entitiy_id, EffectsOnUseComponent).effects

        # Create a new entity with the effects to be processed
        effects_to_be_processed_component = EffectsToBeTargetedComponent(effects=effects)
        manager.create_entity(effects_to_be_processed_component)

        # Remove the active card
        manager.remove_component(entitiy_id, ActiveCardComponent)


# TODO: add effect source to trigger additional effects (e.g., thorns)
class TargetEffectsSystem(BaseSystem):
    def __call__(self, manager: ECSManager) -> None:
        effects_to_be_applied_by_entity: dict[int, list[Effect]] = defaultdict(list)
        for entity_id, effects_to_be_processed_component in manager.get_component(
            EffectsToBeTargetedComponent
        ):
            for effect in effects_to_be_processed_component.effects:
                # Get query target entities
                query_target_entites = [
                    query_entity_id
                    for query_entity_id, _ in manager.get_components(*effect.query_components)
                ]
                # Resolve target entity
                if effect.selection_type == SelectionType.NONE:
                    if len(query_target_entites) > 1:
                        raise ValueError("Too many entities to apply effect")

                    target_entity_id = query_target_entites[0]
                    effects_to_be_applied_by_entity[target_entity_id].append(effect)

                elif effect.selection_type == SelectionType.SPECIFIC:
                    target_entity_id, _ = list(manager.get_component(TargetComponent))[0]
                    if target_entity_id not in query_target_entites:
                        raise ValueError(
                            f"Target entity {target_entity_id} not in query target entities"
                        )

                    effects_to_be_applied_by_entity[target_entity_id].append(effect)

                elif effect.selection_type == SelectionType.RANDOM:
                    target_entity_id = random.choice(query_target_entites)
                    effects_to_be_applied_by_entity[target_entity_id].append(effect)

                elif effect.selection_type == SelectionType.ALL:
                    for target_entity_id in query_target_entites:
                        effects_to_be_applied_by_entity[target_entity_id].append(effect)

            # Remove the effects to be targeted
            # manager.remove_component(entity_id, EffectsToBeTargetedComponent)

        # Add the effects to be applied to the target entities
        for entity_id, effects in effects_to_be_applied_by_entity.items():
            manager.add_component(entity_id, EffectsToBeAppliedComponent(effects=effects))


class ApplyEffectsSystem(BaseSystem):
    def __call__(self, manager: ECSManager) -> None:
        for entity_id, effects_to_be_applied_component in manager.get_component(
            EffectsToBeAppliedComponent
        ):
            for effect in effects_to_be_applied_component.effects:
                _effect_pipeline(manager, entity_id, effect)

            # Remove the effects to be applied
            # manager.remove_component(entity_id, EffectsToBeAppliedComponent)
