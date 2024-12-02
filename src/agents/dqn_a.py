import torch
import torch.nn as nn

from src.agents.base import BaseAgent
from src.agents.dqn.utils import action_idx_to_action
from src.agents.dqn.utils import get_valid_action_mask
from src.game.combat.action import Action
from src.game.combat.view import CombatView


class DQNAgent(BaseAgent):
    def __init__(self, model: nn.Module):
        self._model = model

    @property
    def model(self) -> nn.Module:
        return self._model

    def select_action(self, combat_view: CombatView) -> Action:
        # Set model to evaluation mode TODO: maybe shouldn't be here
        self._model.eval()

        # Calculate q-values for every action
        with torch.no_grad():
            q_t = self.model([combat_view])  # TODO: fix, can sometimes be NaN and cause issues

        # Calculate action w/ highest q-value (masking invalid actions)
        valid_action_mask = get_valid_action_mask(combat_view)
        action_idx = (
            (q_t - (1 - torch.tensor(valid_action_mask, dtype=torch.int)) * 1e20).argmax().item()
        )

        return action_idx_to_action(action_idx, combat_view)
