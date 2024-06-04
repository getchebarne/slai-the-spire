from src.game.ecs.components.common import NameComponent
from src.game.ecs.components.creatures import BlockComponent
from src.game.ecs.components.creatures import CharacterComponent
from src.game.ecs.components.creatures import HealthComponent
from src.game.ecs.components.creatures import MonsterComponent
from src.game.ecs.components.creatures import MonsterIsTurnComponent
from src.game.ecs.components.creatures import MonsterMoveComponent
from src.game.ecs.components.effects import DealDamageEffectComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.components.effects import EffectSelectionType
from src.game.ecs.components.effects import EffectSelectionTypeComponent
from src.game.ecs.components.effects import GainBlockEffectComponent
from src.game.ecs.manager import ECSManager


def create_dummy(manager: ECSManager) -> int:
    base_health = 30

    # Create an attack move
    base_damage = 5
    manager.create_entity(
        MonsterMoveComponent(
            name="Attack",
            effect_entity_ids=[
                manager.create_entity(
                    DealDamageEffectComponent(base_damage),
                    EffectQueryComponentsComponent(CharacterComponent),
                    EffectSelectionTypeComponent(EffectSelectionType.NONE),
                ),
            ],
        )
    )
    # Create a defend move
    base_block = 5
    manager.create_entity(
        MonsterMoveComponent(
            name="Defend",
            effect_entity_ids=[
                manager.create_entity(
                    GainBlockEffectComponent(base_block),
                    EffectQueryComponentsComponent(MonsterIsTurnComponent),
                    EffectSelectionTypeComponent(EffectSelectionType.NONE),
                ),
            ],
        )
    )
    return manager.create_entity(
        MonsterComponent(0),
        NameComponent("Dummy"),
        HealthComponent(base_health),
        BlockComponent(),
    )
