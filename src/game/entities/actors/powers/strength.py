from game.entities.actors.powers.base import BasePower
from game.entities.actors.powers.base import StackType


MIN = -999


class Strength(BasePower):
    def __init__(self, stack_amount: int = 0):
        self.stack_amount = stack_amount

    @property
    def stack_type(self) -> StackType:
        return StackType(intensity=True)

    @property
    def min(self) -> int:
        return MIN
