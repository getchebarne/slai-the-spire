from abc import ABC, abstractmethod


class BaseAI(ABC):
    @abstractmethod
    def next_move_name(self, current_move_name: str) -> str:
        raise NotImplementedError
