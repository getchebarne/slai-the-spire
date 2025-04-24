from enum import Enum
from typing import TypeAlias


CardUpgraded: TypeAlias = bool
AscensionLevel: TypeAlias = int
Floor: TypeAlias = int


class CombatResult(Enum):
    WIN = "WIN"
    LOSS = "LOSS"
