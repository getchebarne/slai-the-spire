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


class MHABlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_hidden_dim: int):
        super().__init__()

        # Multi-head attention layer
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

        # Layer normalization layers
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim), nn.ReLU(), nn.Linear(ff_hidden_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-head attention with residual connection
        x_mha, _ = self.mha(x, x, x, need_weights=False)
        x = self.ln_1(x + x_mha)

        # Feedforward with residual connection
        x_ffn = self.ffn(x)
        x = self.ln_2(x + x_ffn)

        return x


class EmbeddingMLP(nn.Module):
    def __init__(self, linear_sizes: list[int]):
        super().__init__()

        self._linear_sizes = linear_sizes

        self._d_card = 32

        # Modules
        self._enc_card = nn.Linear(7, self._d_card)
        self._pos = nn.Embedding(5, self._d_card)

        self._mha_1 = MHABlock(self._d_card, self._d_card // 8, 32)
        self._mha_2 = MHABlock(self._d_card, self._d_card // 8, 32)

        self._active = nn.Parameter(torch.randn(self._d_card))

        self._ln_3 = nn.LayerNorm(self._d_card * 10)
        self._enc_other = nn.Linear(54, self._d_card * 5)

        self._mlp = MLP(linear_sizes)
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

        # Active mask
        x_mask = x[:, :MAX_HAND_SIZE]

        # Cards
        x_hand = x[:, MAX_HAND_SIZE : MAX_HAND_SIZE + MAX_HAND_SIZE * 7].view(
            batch_size, MAX_HAND_SIZE, 7
        )

        # Other
        x_other = x[:, MAX_HAND_SIZE + MAX_HAND_SIZE * 7 :]

        # Sum positional encodings
        x_hand = self._enc_card(x_hand)
        pos_idxs = torch.arange(5).expand(batch_size, 5).to(x.device)
        pos_embs = self._pos(pos_idxs)
        x_hand += pos_embs

        # MHA
        x_hand_mha = self._mha_1(x_hand)
        x_hand_mha = self._mha_2(x_hand_mha)

        # Active embedding
        x_hand_mha += self._active.unsqueeze(0).unsqueeze(0) * x_mask.unsqueeze(-1)

        # Final hand embedding
        x_hand = torch.flatten(x_hand_mha, start_dim=1)

        # Other
        x_other = self._enc_other(x_other)

        # Concat
        x_hid = self._ln_3(torch.cat([x_hand, x_other], dim=1))

        # Process through MLP
        x = self._mlp(x_hid)
        x = self._last(x)

        return x
