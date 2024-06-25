from src.game.ecs.components.actors import MonsterComponent
from src.game.ecs.components.cards import CardIsPlayedSingletonComponent
from src.game.ecs.components.cards import CardStrikeComponent
from src.game.ecs.components.cards import CardTargetComponent
from src.game.ecs.components.effects import EffectDealDamageComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.utils import add_effect_to_bot


DAMAGE = 6


class CardStrikeSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        query_result = list(
            manager.get_components(CardStrikeComponent, CardIsPlayedSingletonComponent)
        )
        if query_result:
            add_effect_to_bot(
                manager,
                manager.create_entity(
                    EffectDealDamageComponent(DAMAGE),
                    EffectQueryComponentsComponent([MonsterComponent, CardTargetComponent]),
                ),
            )
