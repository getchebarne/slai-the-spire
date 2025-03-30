import torch
import torch.nn as nn
import torch.nn.functional as F

from src.game.combat.constant import MAX_HAND_SIZE
from src.game.combat.constant import MAX_MONSTERS


class MLP(nn.Module):
    def __init__(self, layer_sizes: list[int], bias: bool = True, activation_name: str = "ReLU"):
        super().__init__()

        self._layer_sizes = layer_sizes
        self._bias = bias
        self._activation_name = activation_name

        self._mlp = self._create()

    def _create(self) -> nn.Module:
        layers = []
        for i in range(len(self._layer_sizes) - 1):
            layers.append(nn.Linear(self._layer_sizes[i], self._layer_sizes[i + 1], self._bias))
            layers.append(getattr(nn, self._activation_name)())

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._mlp(x)


class ActorCritic(nn.Module):
    def __init__(
        self,
        layer_sizes_shared: list[int],
        layer_sizes_actor: list[int],
        layer_sizes_critic: list[int],
    ):
        super().__init__()

        self._layer_sizes_shared = layer_sizes_shared
        self._layer_sizes_actor = layer_sizes_actor
        self._layer_sizes_critic = layer_sizes_critic

        self._mlp_shared = MLP(layer_sizes_shared)
        self._mlp_actor = nn.Sequential(
            MLP(layer_sizes_actor),
            nn.Linear(layer_sizes_actor[-1], 2 * MAX_HAND_SIZE + MAX_MONSTERS + 1),
        )
        self._mlp_critic = nn.Sequential(
            MLP(layer_sizes_critic), nn.Linear(layer_sizes_critic[-1], 1)
        )

    def forward(
        self, x_state: torch.Tensor, x_valid_action_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_shared = self._mlp_shared(x_state)

        # Actor
        log_probs = self._mlp_actor(x_shared)
        log_probs_mask = log_probs.masked_fill(x_valid_action_mask == 0, float("-inf"))
        probs = F.softmax(log_probs_mask, dim=-1)

        # Critic
        value = self._mlp_critic(x_shared)

        return probs, value
