from typing import Optional

from game.entities.actors.modifiers.base import BaseModifier
from game.entities.actors.modifiers.base import ModifierType
from game.entities.actors.modifiers.base import Stack
from game.entities.actors.modifiers.base import StackType


STACK_MIN = -999
STACK_MAX = 999
STACK_TYPE = StackType(intensity=True)


class Strength(BaseModifier):
    def __init__(self, stack: Optional[Stack] = None):
        super().__init__(stack if stack is not None else Stack(STACK_TYPE, STACK_MIN, STACK_MAX))

    @property
    def type(self) -> ModifierType:
        if self.stack.amount < 0:
            return ModifierType.DEBUFF

        return ModifierType.BUFF
