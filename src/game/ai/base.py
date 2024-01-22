from abc import ABC, abstractmethod


class BaseAI(ABC):
    def __init__(self):
        self.previous_move_names: list[str] = []

    @abstractmethod
    def next_move_name(self) -> str:
        raise NotImplementedError