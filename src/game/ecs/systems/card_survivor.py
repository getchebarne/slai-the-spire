from src.game.ecs.components.actors import CharacterComponent
from src.game.ecs.components.cards import CardInHandComponent
from src.game.ecs.components.cards import CardIsPlayedComponent
from src.game.ecs.components.cards import CardSurvivorComponent
from src.game.ecs.components.effects import EffectDiscardCardComponent
from src.game.ecs.components.effects import EffectGainBlockComponent
from src.game.ecs.components.effects import EffectNumberOfTargetsComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.components.effects import EffectSelectionType
from src.game.ecs.components.effects import EffectSelectionTypeComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.utils import add_effect_to_bot


BLOCK = 8
DISCARD = 1


class CardSurvivorSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        query_result = list(manager.get_components(CardSurvivorComponent, CardIsPlayedComponent))
        if query_result:
            add_effect_to_bot(
                manager,
                manager.create_entity(
                    EffectGainBlockComponent(BLOCK),
                    EffectQueryComponentsComponent([CharacterComponent]),
                ),
            )
            add_effect_to_bot(
                manager,
                manager.create_entity(
                    EffectDiscardCardComponent(),
                    EffectQueryComponentsComponent([CardInHandComponent]),
                    EffectSelectionTypeComponent(EffectSelectionType.SPECIFIC),
                    EffectNumberOfTargetsComponent(DISCARD),
                ),
            )
