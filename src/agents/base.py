from abc import ABC, abstractmethod

from src.game.combat.action import Action
from src.game.combat.view import CombatView


class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, combat_view: CombatView) -> Action:
        raise NotImplementedError
