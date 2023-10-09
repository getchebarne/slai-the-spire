from dataclasses import dataclass

from components.base import BaseComponent


@dataclass
class Health(BaseComponent):
    current: int
    max: int
