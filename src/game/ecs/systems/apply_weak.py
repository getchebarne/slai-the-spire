from src.game.ecs.components.actors import ModifierParentComponent
from src.game.ecs.components.actors import ModifierWeakComponent
from src.game.ecs.components.effects import EffectDealDamageComponent
from src.game.ecs.components.effects import EffectIsTargetedComponent
from src.game.ecs.components.effects import EffectParentComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


class ApplyWeakSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        query_result = list(
            manager.get_components(EffectDealDamageComponent, EffectIsTargetedComponent)
        )
        if query_result:

            effect_entity_id, (effect_deal_damage_component, _) = query_result[0]

            effect_parent_component = manager.get_component_for_entity(
                effect_entity_id, EffectParentComponent
            )
            if effect_parent_component is not None:
                for _, (_, modifier_parent_component) in manager.get_components(
                    ModifierWeakComponent, ModifierParentComponent
                ):
                    if (
                        modifier_parent_component.actor_entity_id
                        == effect_parent_component.entity_id
                    ):
                        # Source is weakened
                        effect_deal_damage_component.value *= 0.75
