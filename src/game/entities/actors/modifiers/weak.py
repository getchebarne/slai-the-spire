from typing import Optional

from game.entities.actors.modifiers.base import BaseModifier
from game.entities.actors.modifiers.base import ModifierType
from game.entities.actors.modifiers.base import Stack
from game.entities.actors.modifiers.base import StackType


STACK_MIN = 0
STACK_MAX = 999
STACK_TYPE = StackType(duration=True)


class Weak(BaseModifier):
    def __init__(self, stack: Optional[Stack] = None):
        super().__init__(stack if stack is not None else Stack(STACK_TYPE, STACK_MIN, STACK_MAX))

    @property
    def type(self) -> ModifierType:
        return ModifierType.DEBUFF
