import torch
import torch.nn as nn

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
        for i in range(len(self._layer_sizes)):
            if i == 0:
                # First layer is lazy
                layers.append(nn.LazyLinear(self._layer_sizes[i], self._bias))
            else:
                layers.append(
                    nn.Linear(self._layer_sizes[i - 1], self._layer_sizes[i], self._bias)
                )

            layers.append(getattr(nn, self._activation_name)())

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._mlp(x)


class EmbeddingMLP(nn.Module):
    def __init__(self, linear_sizes: list[int]):
        super().__init__()

        self._linear_sizes = linear_sizes

        # Modules
        self._mlp = MLP(linear_sizes)
        self._last = nn.Linear(linear_sizes[-1], MAX_HAND_SIZE + MAX_MONSTERS + 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process through MLP
        x = self._mlp(x)
        x = self._last(x)

        return x
