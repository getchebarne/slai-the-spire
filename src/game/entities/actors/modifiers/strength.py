from game.entities.actors.modifiers.base import BaseModifier
from game.entities.actors.modifiers.base import ModifierType
from game.entities.actors.modifiers.base import Stack
from game.entities.actors.modifiers.base import StackType


MIN = -999
STACK_TYPE = StackType(intensity=True)


class Strength(BaseModifier):
    def __init__(self, stack: Stack = Stack(STACK_TYPE, amount=0, min=MIN)):
        super().__init__(stack)

    @property
    def type(self) -> ModifierType:
        if self.stack.amount < 0:
            return ModifierType.DEBUFF

        return ModifierType.BUFF
