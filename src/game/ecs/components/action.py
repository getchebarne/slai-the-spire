from dataclasses import dataclass

from src.game.ecs.components.base import BaseComponent


@dataclass
class ActionComponent(BaseComponent):
    pass


@dataclass
class ActionSelectComponent(BaseComponent):
    pass


@dataclass
class ActionConfirmComponent(BaseComponent):
    pass


@dataclass
class ActionEndTurnComponent(BaseComponent):
    pass
