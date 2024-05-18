from abc import ABC, abstractmethod

from src.game.ecs.manager import ECSManager


class BaseSystem(ABC):
    @abstractmethod
    def __call__(self, manager: ECSManager) -> None:
        pass
