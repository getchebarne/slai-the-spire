import torch
import torch.nn as nn
import torch.nn.functional as F

from src.game.combat.constant import MAX_HAND_SIZE
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
        for i in range(len(self._layer_sizes) - 1):
            layers.append(nn.Linear(self._layer_sizes[i], self._layer_sizes[i + 1], self._bias))
            layers.append(getattr(nn, self._activation_name)())

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._mlp(x)


class HandTransformer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int):
        super().__init__()

        self._mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self._ln_1 = nn.LayerNorm(embed_dim)
        self._ln_2 = nn.LayerNorm(embed_dim)
        self._ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim)
        )

    def _mha_with_fallback(self, x_hand: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        # x_hand: (B, MAX_HAND_SIZE, D)
        # x_mask: (B, MAX_HAND_SIZE)
        x_empty = torch.all(x_mask, dim=1)  # (B,)

        if torch.all(x_empty):
            # Batch full of empty hands. Should be very rare
            return torch.zeros_like(x_hand)

        if not torch.any(x_empty):
            # No empty hands in batch. Should be fairly common
            return self._mha(x_hand, x_hand, x_hand, key_padding_mask=x_mask, need_weights=False)[
                0
            ]

        # Mix of empty and non-empty hands
        x_non_empty = ~x_empty
        x_hand_non_empty = x_hand[x_non_empty]
        x_mask_non_empty = x_mask[x_non_empty]

        # Process non-empty hands
        x_out_non_empty, _ = self._mha(
            x_hand_non_empty,
            x_hand_non_empty,
            x_hand_non_empty,
            key_padding_mask=x_mask_non_empty,
            need_weights=False,
        )

        # Create empty (zeroed) output tensor and fill non-empty hands w/ MHA output
        x_out = torch.zeros_like(x_hand)
        x_out[x_non_empty] = x_out_non_empty

        # (B, MAX_HAND_SIZE, embed_dim)
        return x_out

    def forward(self, x_hand: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        # Multi-head attention
        x_mha = self._mha_with_fallback(x_hand, x_mask)
        x = self._ln_1(x_hand + x_mha)

        # Feedforward with residual connection
        x_ff = self._ff(x)
        x = self._ln_2(x + x_ff)

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
    def __init__(self, dim_card: int, attn_card_num_heads: int, linear_sizes: list[int]):
        super().__init__()

        self._dim_card = dim_card
        self._attn_card_num_heads = attn_card_num_heads
        self._linear_sizes = linear_sizes

        # Modules
        self._enc_card = nn.Linear(2 + NUM_EFFECTS, self._dim_card)
        self._hand_transformer_1 = HandTransformer(
            self._dim_card, self._attn_card_num_heads, self._dim_card
        )
        self._hand_transformer_2 = HandTransformer(
            self._dim_card, self._attn_card_num_heads, self._dim_card
        )
        self._enc_other = ResidualMLP(30, self._dim_card)

        self._ln_common = nn.LayerNorm(self._dim_card * 5)
        self._mlp_common = MLP([self._dim_card * 5] + linear_sizes)

        self._ln_hand = nn.LayerNorm(self._linear_sizes[-1] + self._dim_card)
        self._mlp_hand = MLP([self._linear_sizes[-1] + self._dim_card] + linear_sizes)

        self._ln_monster = nn.LayerNorm(self._linear_sizes[-1] + self._dim_card)
        self._mlp_monster = MLP([self._linear_sizes[-1] + self._dim_card] + linear_sizes)

        self._last_hand = nn.Linear(linear_sizes[-1], 2)
        self._last_monster = nn.Linear(linear_sizes[-1], 1)
        self._last_end = nn.Linear(linear_sizes[-1], 1)

    # TODO: improve unpacking
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Hand size
        high = 1
        x_hand_size = x[:, :high]
        x_hand_mask = torch.arange(MAX_HAND_SIZE).unsqueeze(0) < x_hand_size  # (B, C)
        x_hand_mask = x_hand_mask.view(batch_size, MAX_HAND_SIZE, 1)  # (B, C, 1)

        # Active mask
        low = high
        high += MAX_HAND_SIZE
        x_active_mask = x[:, low:high]

        # Hand
        low = high
        high += MAX_HAND_SIZE * (2 + NUM_EFFECTS)
        x_cards = x[:, low:high].to(torch.float32).view(batch_size, MAX_HAND_SIZE, 2 + NUM_EFFECTS)

        # Other
        low = high
        x_other = x[:, low:].to(torch.float32)

        # Get card encodings
        x_hand = self._enc_card(x_cards) * x_hand_mask  # (B, L, D)
        x_hand_mha = self._hand_transformer_1(x_hand, ~x_hand_mask.view(batch_size, MAX_HAND_SIZE))
        x_hand_mha = self._hand_transformer_2(
            x_hand_mha, ~x_hand_mask.view(batch_size, MAX_HAND_SIZE)
        )

        # Other
        x_other = self._enc_other(x_other)

        x_hand_mha_mask = x_hand_mha * x_hand_mask

        # Global state embedding
        x_global = torch.cat(
            [
                torch.mean(x_hand_mha_mask, dim=1),
                torch.sum(x_hand_mha_mask, dim=1),
                torch.max(x_hand_mha_mask, dim=1)[0],
                torch.min(x_hand_mha_mask, dim=1)[0],
                x_other,
            ],
            dim=1,
        )
        x_global = self._mlp_common(self._ln_common(x_global))

        # Process cards in hand
        x_hand = torch.cat(
            [
                x_global.unsqueeze(1).expand(batch_size, MAX_HAND_SIZE, self._linear_sizes[-1]),
                x_hand_mha_mask,
            ],
            dim=2,
        )
        x_hand = self._mlp_hand(self._ln_hand(x_hand)) * x_hand_mask

        # Monster
        x_active = torch.sum(x_hand_mha * x_active_mask.unsqueeze(-1), dim=1)
        x_monster = self._mlp_monster(self._ln_monster(torch.cat([x_global, x_active], dim=1)))

        # Final action values
        x_hand = self._last_hand(x_hand)
        x_monster = self._last_monster(x_monster)
        x_end = self._last_end(x_global)

        x = torch.cat(
            [x_hand.permute(0, 2, 1).flatten(start_dim=1), x_monster, x_end],
            dim=1,
        )

        return x
