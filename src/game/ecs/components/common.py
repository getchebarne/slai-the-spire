from dataclasses import dataclass

from src.game.ecs.components.base import BaseComponent


@dataclass
class NameComponent(BaseComponent):
    value: str
