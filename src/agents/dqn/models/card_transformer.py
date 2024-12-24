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
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )

        # Layer normalization layers
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim), nn.ReLU(), nn.Linear(ff_hidden_dim, embed_dim)
        )

        self.pad_embedding = nn.Parameter(torch.randn(ff_hidden_dim))

    def masked_attention(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        fully_masked = mask.all(dim=1)  # Shape: (B,)

        if fully_masked.all():
            # If all sequences are fully masked, return all zeros
            return torch.zeros_like(x)

        if not fully_masked.any():
            # If no sequences are fully masked, process normally
            output, _ = self.mha(x, x, x, key_padding_mask=mask, need_weights=False)
            return output

        # For mixed case, only process unmasked sequences
        unmasked_indices = ~fully_masked

        # Select only unmasked sequences and their masks
        x_unmasked = x[unmasked_indices]
        mask_unmasked = mask[unmasked_indices]

        # Process unmasked sequences
        output_unmasked, _ = self.mha(
            x_unmasked, x_unmasked, x_unmasked, key_padding_mask=mask_unmasked, need_weights=False
        )

        # Create output tensor
        output = torch.zeros_like(x)
        output[unmasked_indices] = output_unmasked

        return output

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Multi-head attention
        x_mha = self.masked_attention(x, mask)
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
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.fc3 = nn.Linear(output_dim, output_dim)

        # Layer normalization for the residual connection
        self.ln = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project the input features to `output_dim` & save residual
        x = self.fc1(x)
        x_res = x

        # Process features w/ two-layer MLP
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)

        # Make residual connection and layer-normalize
        x = self.ln(x + x_res)

        return x


class CardTransformer(nn.Module):
    def __init__(self, linear_sizes: list[int]):
        super().__init__()

        self._linear_sizes = linear_sizes

        self._d_card = 48

        # Modules
        self._enc_card = nn.Linear(NUM_EFFECTS, self._d_card)
        # self._cost = nn.Embedding(4, self._d_card)
        # self._energy = nn.Embedding(4, self._d_card)
        self._energy = nn.Linear(1, self._d_card)

        self._mha_1 = MHABlock(self._d_card, self._d_card // 8, self._d_card)

        self._active = nn.Parameter(torch.randn(self._d_card))

        self._enc_other = ResidualMLP(29, self._d_card * MAX_HAND_SIZE // 2)
        self._ln_3 = nn.LayerNorm(
            self._d_card * MAX_HAND_SIZE + self._d_card * MAX_HAND_SIZE // 2 + self._d_card
        )

        self._mlp = MLP(linear_sizes)
        self._last = nn.Linear(linear_sizes[-1], 2 * MAX_HAND_SIZE + MAX_MONSTERS + 1)

        # padding
        self._padding = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Hand size
        high = 1
        x_size = x[:, :high]
        x_size = torch.arange(MAX_HAND_SIZE).unsqueeze(0) < x_size  # (B, C)
        x_size = x_size.view(batch_size, MAX_HAND_SIZE, 1)  # (B, C, 1)

        # Active mask
        low = high
        high += MAX_HAND_SIZE
        x_mask = x[:, low:high]

        # Costs + MAX_HAND_SIZE 1 cost x card
        low = high
        high += MAX_HAND_SIZE
        x_costs = x[:, low:high].view(batch_size, MAX_HAND_SIZE)  # (B, C)

        # Cards + MAX_HAND_SIZE * 5 effect x card
        low = high
        NUM_EFFECTS = 5
        high += MAX_HAND_SIZE * NUM_EFFECTS
        x_cards = x[:, low:high].to(torch.float32).view(batch_size, MAX_HAND_SIZE, NUM_EFFECTS)

        # Energy
        low = high
        high += 1
        x_energy = x[:, low:high]  # (B, 1)

        # Energy - Cost
        x_energy_cost = (
            self._energy((x_energy - x_costs).view(batch_size, MAX_HAND_SIZE, 1).to(torch.float32))
            * x_size
        )

        # Other
        low = high
        x_other = x[:, low:].to(torch.float32)

        # Sum positional encodings
        # pos_idxs = torch.arange(MAX_HAND_SIZE, device=x.device)  # (B, D)
        # pos_embs = self._pos(pos_idxs).unsqueeze(0)  # (B, 1, D)
        # x_hand += pos_embs  # (B, L, D)

        # Get card encodings
        x_hand = self._enc_card(x_cards) * x_size  # (B, L, D)

        # Sum energy encoding
        x_hand += x_energy_cost

        # MHA
        x_hand_mha = self._mha_1(x_hand, ~x_size.view(batch_size, MAX_HAND_SIZE))
        # x_hand_mha = self._mha_2(x_hand_mha)

        # Active embedding
        x_hand_mha += self._active.unsqueeze(0).unsqueeze(0) * x_mask.unsqueeze(-1)

        # Final hand embedding
        x_hand_global = torch.mean(x_hand_mha * x_size, dim=1)
        x_hand = torch.flatten(x_hand_mha, start_dim=1)

        # Other
        x_other = self._enc_other(x_other)

        # Concat
        x_hid = self._ln_3(torch.cat([x_hand, x_hand_global, x_other], dim=1))

        # Process through MLP
        x = self._mlp(x_hid)
        x = self._last(x)

        return x
