from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.systems.effect import DealDamageSystem
from src.game.ecs.systems.effect import GainBlockSystem
from src.game.ecs.systems.engine import PlayCardSystem
from src.game.ecs.systems.engine import TargetEffectsSystem


def create_systems() -> list[BaseSystem]:
    return [PlayCardSystem(), TargetEffectsSystem(), GainBlockSystem(), DealDamageSystem()]
