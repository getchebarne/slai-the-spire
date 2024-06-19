from src.game.ecs.components.actors import CharacterComponent
from src.game.ecs.components.cards import CardDefendComponent
from src.game.ecs.components.cards import CardIsPlayedComponent
from src.game.ecs.components.effects import EffectGainBlockComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.utils import add_effect_to_bot


BLOCK = 5


class CardDefendSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        query_result = list(manager.get_components(CardDefendComponent, CardIsPlayedComponent))
        if query_result:
            add_effect_to_bot(
                manager,
                manager.create_entity(
                    EffectGainBlockComponent(BLOCK),
                    EffectQueryComponentsComponent([CharacterComponent]),
                ),
            )
