import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.dqn.encoder import MAX_LEN_DISC_PILE
from src.agents.dqn.encoder import MAX_LEN_DRAW_PILE
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

        self._d_card = 24

        # Modules
        self._mlp = MLP(linear_sizes)
        self._enc_card_1 = nn.Linear(7, self._d_card)
        self._enc_card_2 = nn.Linear(self._d_card, self._d_card)
        self._enc_card_3 = nn.Linear(self._d_card, self._d_card)
        self._pos = nn.Embedding(5, self._d_card)
        self._mha = nn.MultiheadAttention(self._d_card, num_heads=3, batch_first=True)
        self._ln_1 = nn.LayerNorm(self._d_card)
        self._ln_2 = nn.LayerNorm(self._d_card)
        self._ln_3 = nn.LayerNorm(self._d_card * 10)
        self._enc_other = nn.Linear(27, self._d_card * 5)
        self._last = nn.Linear(linear_sizes[-1], 2 * MAX_HAND_SIZE + MAX_MONSTERS + 1, bias=True)

        # LayerNorms
        # self._ln_hand = nn.LayerNorm(48)
        # self._ln_draw = nn.LayerNorm(48)
        # self._ln_disc = nn.LayerNorm(48)
        # self._ln_cards = nn.LayerNorm(48 * 7)
        # self._ln_other = nn.LayerNorm(48 * 2)
        # self._ln_hid = nn.LayerNorm(48 * 9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Cards
        x_hand = x[:, : MAX_HAND_SIZE * 7].view(batch_size, MAX_HAND_SIZE, 7)

        # Cards linear encoding + pos encoding
        x_hand = self._enc_card_1(x_hand)

        # Sum positional encodings
        pos_idxs = torch.arange(5).expand(batch_size, 5).to(x.device)
        pos_embs = self._pos(pos_idxs)
        x_hand += pos_embs

        # MHA
        x_hand_mha, _ = self._mha(x_hand, x_hand, x_hand, need_weights=False)

        # First residual
        x_hand = self._ln_1(x_hand + x_hand_mha)

        # Second residual
        x_hand = self._ln_2(x_hand + self._enc_card_3(F.relu(self._enc_card_2(x_hand))))

        # Flatten
        x_hand = torch.flatten(x_hand, start_dim=1)

        # Flatten hand
        x_hand = torch.flatten(x_hand, start_dim=1)

        # Other
        x_other = x[:, MAX_HAND_SIZE * 7 :]
        x_other = self._enc_other(x_other)

        # Concat
        x_hid = self._ln_3(torch.cat([x_hand, x_other], dim=1))

        # Process through MLP
        x = self._mlp(x_hid)
        x = self._last(x)

        return x
