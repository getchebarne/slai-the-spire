from dataclasses import dataclass
from typing import Optional


@dataclass
class Energy:
    max: int = 3
    current: Optional[int] = None

    def __post_init__(self) -> None:
        if self.current is None:
            self.current = self.max

        elif self.current > self.max:
            raise ValueError("Current energy can't be larger than maximum energy")
