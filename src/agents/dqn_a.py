import torch
import torch.nn as nn

from src.agents.base import BaseAgent
from src.game.combat.view import CombatView


class DQNAgent(BaseAgent):
    def __init__(self, model: nn.Module):
        self._model = model

    @property
    def model(self) -> nn.Module:
        return self._model

    # TODO: revisit if `valid_action_mask` should be necessary here
    def select_action(self, combat_view: CombatView, valid_action_mask: list[bool]) -> int:
        with torch.no_grad():
            q_t = self.model([combat_view])

        # Calculate action w/ highest q-value (masking invalid actions)
        action_idx = (
            (q_t - (1 - torch.tensor(valid_action_mask, dtype=torch.int)) * 1e20).argmax().item()
        )

        return action_idx
