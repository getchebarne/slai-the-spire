from dataclasses import dataclass


from components.base import BaseComponent


MAX_BLOCK = 999


@dataclass
class Block(BaseComponent):
    current: int
    max: int = MAX_BLOCK
