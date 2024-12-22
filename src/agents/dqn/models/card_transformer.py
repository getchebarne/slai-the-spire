import torch
import torch.nn as nn
import torch.nn.functional as F

from src.game.combat.constant import MAX_HAND_SIZE
from src.game.combat.constant import MAX_MONSTERS
from src.game.combat.constant import NUM_EFFECTS


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


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim

        # Fully connected layers
        self._fc_1 = nn.Linear(input_dim, output_dim)
        self._fc_2 = nn.Linear(output_dim, output_dim)
        self._fc_3 = nn.Linear(output_dim, output_dim)

        # Layer normalization for the residual connection
        self._ln = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project the input features to `output_dim` & save residual
        x = self._fc_1(x)
        x_res = x

        # Process features w/ two-layer MLP
        x = self._fc_2(x)
        x = F.relu(x)
        x = self._fc_3(x)
        x = F.relu(x)

        # Make residual connection and layer-normalize
        x = self._ln(x + x_res)

        return x


class CardTransformer(nn.Module):
    def __init__(self, linear_sizes: list[int]):
        super().__init__()

        self._linear_sizes = linear_sizes

        self._d_card = 48

        # Modules
        self._enc_card = nn.Linear(NUM_EFFECTS, self._d_card)
        self._cost = nn.Embedding(4, self._d_card)
        self._energy = nn.Embedding(4, self._d_card)

        self._mha_1 = MHABlock(self._d_card, self._d_card // 8, self._d_card)
        # self._mha_2 = MHABlock(self._d_card, self._d_card // 8, 32)

        self._active = nn.Parameter(torch.randn(self._d_card))

        self._enc_other = ResidualMLP(29, self._d_card * MAX_HAND_SIZE // 2)
        self._ln_3 = nn.LayerNorm(self._d_card * MAX_HAND_SIZE + self._d_card * MAX_HAND_SIZE // 2)

        self._mlp = MLP(linear_sizes)
        self._last = nn.Linear(linear_sizes[-1], 2 * MAX_HAND_SIZE + MAX_MONSTERS + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Active mask
        high = MAX_HAND_SIZE
        x_mask = x[:, :high]

        # Costs + MAX_HAND_SIZE 1 cost x card
        low = high
        high += MAX_HAND_SIZE
        x_costs = x[:, low:high].view(batch_size, MAX_HAND_SIZE)
        x_costs = self._cost(x_costs)

        # Cards + MAX_HAND_SIZE * 5 effect x card
        low = high
        NUM_EFFECTS = 5
        high += MAX_HAND_SIZE * NUM_EFFECTS
        x_cards = x[:, low:high].to(torch.float32).view(batch_size, MAX_HAND_SIZE, NUM_EFFECTS)

        # Energy
        low = high
        high += 1
        x_energy = x[:, low:high]  # (B, D)
        x_energy = self._energy(x_energy)

        # Other
        low = high
        x_other = x[:, low:].to(torch.float32)

        # Sum positional encodings
        # pos_idxs = torch.arange(MAX_HAND_SIZE, device=x.device)  # (B, D)
        # pos_embs = self._pos(pos_idxs).unsqueeze(0)  # (B, 1, D)
        # x_hand += pos_embs  # (B, L, D)

        # Get card encodings
        x_hand = self._enc_card(x_cards)  # (B, L, D)

        # Sum energy encoding
        x_hand += x_energy

        # Sum cost encodings
        x_hand += x_costs

        # MHA
        x_hand_mha = self._mha_1(x_hand)
        # x_hand_mha = self._mha_2(x_hand_mha)

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
