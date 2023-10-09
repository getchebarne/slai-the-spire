from dataclasses import dataclass

from components.base import BaseComponent


@dataclass
class Effect(BaseComponent):
    name: str
    value: int
