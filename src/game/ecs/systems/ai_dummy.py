from src.game.ecs.components.creatures import CharacterComponent
from src.game.ecs.components.creatures import DummyAIComponent
from src.game.ecs.components.creatures import IsTurnComponent
from src.game.ecs.components.creatures import MonsterComponent
from src.game.ecs.components.creatures import MonsterMoveComponent
from src.game.ecs.components.creatures import MonsterPendingMoveUpdateComponent
from src.game.ecs.components.effects import EffectDealDamageComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.components.effects import EffectSelectionType
from src.game.ecs.components.effects import EffectSelectionTypeComponent
from src.game.ecs.components.effects import EffectGainBlockComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


# TODO: improve how moves are defined
class AIDummySystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        try:
            monster_entity_id, _ = next(
                manager.get_components(DummyAIComponent, MonsterPendingMoveUpdateComponent)
            )

        except StopIteration:
            return

        # Get current move
        monster_move_component = manager.get_component_for_entity(
            monster_entity_id, MonsterMoveComponent
        )

        if monster_move_component is None:
            manager.add_component(
                monster_entity_id,
                MonsterMoveComponent(
                    "Attack",
                    [
                        manager.create_entity(
                            EffectDealDamageComponent(0),
                            EffectQueryComponentsComponent([CharacterComponent]),
                            EffectSelectionTypeComponent(EffectSelectionType.NONE),
                        )
                    ],
                ),
            )

        else:
            if monster_move_component.name == "Attack":
                monster_move_component.name = "Defend"
                monster_move_component.effect_entity_ids = [
                    manager.create_entity(
                        EffectGainBlockComponent(100),
                        EffectQueryComponentsComponent([MonsterComponent, IsTurnComponent]),
                        EffectSelectionTypeComponent(EffectSelectionType.NONE),
                    ),
                ]

            elif monster_move_component.name == "Defend":
                monster_move_component.name = "Attack"
                monster_move_component.effect_entity_ids = [
                    manager.create_entity(
                        EffectDealDamageComponent(0),
                        EffectQueryComponentsComponent([CharacterComponent]),
                        EffectSelectionTypeComponent(EffectSelectionType.NONE),
                    )
                ]

            else:
                raise ValueError(f"Invalid move name: {monster_move_component.name}")

        # Untag
        manager.remove_component(monster_entity_id, MonsterPendingMoveUpdateComponent)
