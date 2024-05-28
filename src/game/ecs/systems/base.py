from abc import ABC, abstractmethod
from enum import Enum

from src.game.ecs.manager import ECSManager


class ProcessStatus(Enum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    PASS = "PASS"


# TODO: add priority
class BaseSystem(ABC):
    @abstractmethod
    def process(self, manager: ECSManager) -> ProcessStatus:
        pass
