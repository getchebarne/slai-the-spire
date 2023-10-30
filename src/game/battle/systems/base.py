from abc import ABC, abstractmethod
from typing import Any


class BaseSystem(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        raise NotImplementedError
