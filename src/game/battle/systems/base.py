from abc import ABC
from abc import abstractmethod
from typing import Any


class BaseSystem(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        raise NotImplementedError
