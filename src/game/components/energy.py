from dataclasses import dataclass


from components.base import BaseComponent


@dataclass
class Energy(BaseComponent):
    current: int
    max: int
