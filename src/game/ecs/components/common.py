from dataclasses import dataclass

from src.game.ecs.components.base import BaseComponent


@dataclass
class NameComponent(BaseComponent):
    value: str


@dataclass
class DescriptionComponent(BaseComponent):
    value: str


@dataclass
class IsSelectedComponent(BaseComponent):
    pass


@dataclass
class CanBeSelectedComponent(BaseComponent):
    pass
