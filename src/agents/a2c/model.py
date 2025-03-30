import torch
import torch.nn as nn
import torch.nn.functional as F

from src.game.combat.constant import MAX_HAND_SIZE
from src.game.combat.constant import NUM_EFFECTS


# TODO: improve this, layer_sizes is not actually layer sizes
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
        dim_card: int,
    ):
        super().__init__()

        self._layer_sizes_shared = layer_sizes_shared
        self._layer_sizes_actor = layer_sizes_actor
        self._layer_sizes_critic = layer_sizes_critic
        self._dim_card = dim_card

        self._embedding_card = nn.Linear(NUM_EFFECTS + 3, dim_card)

        self._last_card = nn.Sequential(
            MLP(
                [
                    layer_sizes_shared[-1] + dim_card,
                    layer_sizes_shared[-1] + dim_card,
                ]
            ),
            nn.Linear(layer_sizes_shared[-1] + dim_card, 2),
        )
        self._last_monster = nn.Sequential(
            MLP(
                [
                    layer_sizes_shared[-1],
                    layer_sizes_shared[-1] // 2,
                ]
            ),
            nn.Linear(layer_sizes_shared[-1] // 2, 1),
        )
        self._last_end_turn = nn.Sequential(
            MLP(
                [
                    layer_sizes_shared[-1],
                    layer_sizes_shared[-1] // 2,
                ]
            ),
            nn.Linear(layer_sizes_shared[-1] // 2, 1),
        )

        self._mlp_shared = MLP(layer_sizes_shared)
        self._mlp_critic = nn.Sequential(
            MLP(layer_sizes_critic), nn.Linear(layer_sizes_critic[-1], 1)
        )

    def forward(
        self, x_state: dict[str, torch.Tensor], x_valid_action_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_card = self._embedding_card(x_state["hand"])
        x_shared = self._mlp_shared(
            torch.cat(
                [
                    torch.flatten(x_card),
                    x_state["character"],
                    x_state["monsters"],
                    x_state["energy"],
                ]
            )
        )
        x_hand = torch.cat(
            [
                x_shared.unsqueeze(0).expand((MAX_HAND_SIZE, self._layer_sizes_shared[-1])),
                x_card,
            ],
            dim=1,
        )

        # Actor
        log_probs_card = self._last_card(x_hand)
        log_probs_monster = self._last_monster(x_shared)
        log_probs_end_turn = self._last_end_turn(x_shared)
        log_probs = torch.cat(
            [
                torch.flatten(log_probs_card.permute(1, 0)),
                log_probs_monster,
                log_probs_end_turn,
            ]
        )
        log_probs_mask = log_probs.masked_fill(x_valid_action_mask == 0, float("-inf"))
        probs = F.softmax(log_probs_mask, dim=-1)

        # Critic
        value = self._mlp_critic(x_shared)

        return probs, value
