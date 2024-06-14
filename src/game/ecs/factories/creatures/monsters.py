from src.game.ecs.components.common import NameComponent
from src.game.ecs.components.creatures import BlockComponent
from src.game.ecs.components.creatures import CharacterComponent
from src.game.ecs.components.creatures import DummyAIComponent
from src.game.ecs.components.creatures import HealthComponent
from src.game.ecs.components.creatures import IsTurnComponent
from src.game.ecs.components.creatures import MonsterComponent
from src.game.ecs.components.creatures import MonsterHasMovesComponent
from src.game.ecs.components.creatures import MonsterMoveComponent
from src.game.ecs.components.creatures import MonsterMoveHasEffectsComponent
from src.game.ecs.components.creatures import MonsterMoveIntentComponent
from src.game.ecs.components.creatures import MonsterPendingMoveUpdateComponent
from src.game.ecs.components.effects import EffectDealDamageComponent
from src.game.ecs.components.effects import EffectGainBlockComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.manager import ECSManager


def create_dummy(manager: ECSManager) -> int:
    base_health = 30

    # Create moves
    base_damage = 5
    attack_move_entity_id = manager.create_entity(
        MonsterMoveComponent("Attack"),
        MonsterMoveHasEffectsComponent(
            [
                manager.create_entity(
                    EffectDealDamageComponent(base_damage),
                    EffectQueryComponentsComponent([CharacterComponent]),
                )
            ]
        ),
        MonsterMoveIntentComponent(damage=base_damage, times=1, block=False),
    )
    base_block = 5
    defend_move_entity_id = manager.create_entity(
        MonsterMoveComponent("Defend"),
        MonsterMoveHasEffectsComponent(
            [
                manager.create_entity(
                    EffectGainBlockComponent(base_block),
                    EffectQueryComponentsComponent([MonsterComponent, IsTurnComponent]),
                )
            ]
        ),
        MonsterMoveIntentComponent(damage=0, times=0, block=True),
    )
    return manager.create_entity(
        MonsterComponent(0),
        NameComponent("Dummy"),
        DummyAIComponent(),
        HealthComponent(base_health),
        BlockComponent(),
        MonsterHasMovesComponent([attack_move_entity_id, defend_move_entity_id]),
        MonsterPendingMoveUpdateComponent(),
    )
