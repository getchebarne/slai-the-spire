from dataclasses import dataclasss
from typing import List


from components.effect import Effect
from entities.base import BaseEntity


# TODO: add `text` attribute
@dataclasss
class Card(BaseEntity):
    name: str
    cost: int
    effects: List[Effect]
