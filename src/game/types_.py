from collections import deque
from typing import TypeAlias

from src.game.core.effect import Effect


CardUpgraded: TypeAlias = bool
AscensionLevel: TypeAlias = int
Floor: TypeAlias = int
EffectQueue: TypeAlias = deque[Effect]
