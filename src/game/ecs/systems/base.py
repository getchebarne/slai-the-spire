from abc import ABC, abstractmethod

from src.game.ecs.manager import ECSManager


# TODO: add priority
class BaseSystem(ABC):
    @abstractmethod
    def process(self, manager: ECSManager) -> None:
        pass
