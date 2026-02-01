import torch
import torch.nn as nn

from src.rl.encoding import get_card_encoding_dim
from src.rl.encoding import get_character_encoding_dim
from src.rl.encoding import get_energy_encoding_dim
from src.rl.encoding import get_monster_encoding_dim
from src.rl.models.mlp import MLP


DIM_ENC_CARD = get_card_encoding_dim()
DIM_ENC_CHARACTER = get_character_encoding_dim()
DIM_ENC_ENERGY = get_energy_encoding_dim()
DIM_ENC_MONSTER = get_monster_encoding_dim()
DIM_ENC_OTHER = DIM_ENC_CHARACTER + DIM_ENC_MONSTER + DIM_ENC_ENERGY


class CardTransformer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int):
        super().__init__()

        self._mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self._ln_1 = nn.LayerNorm(embed_dim)
        self._ln_2 = nn.LayerNorm(embed_dim)
        self._ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim)
        )

    def _mha_with_fallback(self, x_card: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        # x_card: (B, MAX_HAND_SIZE, D)
        # x_mask: (B, MAX_HAND_SIZE)
        x_empty = torch.all(x_mask, dim=1)  # (B,)

        if torch.all(x_empty):
            # Batch full of empty hands. Should be very rare
            return torch.zeros_like(x_card)

        if not torch.any(x_empty):
            # No empty hands in batch. Should be fairly common
            return self._mha(x_card, x_card, x_card, key_padding_mask=x_mask, need_weights=False)[
                0
            ]

        # Mix of empty and non-empty hands
        x_non_empty = ~x_empty
        x_card_non_empty = x_card[x_non_empty]
        x_mask_non_empty = x_mask[x_non_empty]

        # Process non-empty hands
        x_out_non_empty, _ = self._mha(
            x_card_non_empty,
            x_card_non_empty,
            x_card_non_empty,
            key_padding_mask=x_mask_non_empty,
            need_weights=False,
        )

        # Create empty (zeroed) output tensor and fill non-empty hands w/ MHA output
        x_out = torch.zeros_like(x_card)
        x_out[x_non_empty] = x_out_non_empty

        # (B, MAX_HAND_SIZE, embed_dim)
        return x_out

    def forward(self, x_card: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        # Multi-head attention
        x_mha = self._mha_with_fallback(x_card, x_mask)
        x = self._ln_1(x_card + x_mha)

        # Feedforward with residual connection
        x_ff = self._ff(x)
        x = self._ln_2(x + x_ff)

        return x


# TODO: parametrize dimensions better, so they are more readable
class DeepQNetwork(nn.Module):
    def __init__(self, dim_card: int, dim_char: int, dim_monster: int, dim_energy: int):
        super().__init__()

        self._dim_card = dim_card
        self._dim_char = dim_char
        self._dim_monster = dim_monster
        self._dim_energy = dim_energy

        # Card embedding
        self._embedding_card = nn.Linear(DIM_ENC_CARD, dim_card)

        # Character embedding
        self._embedding_char = MLP([DIM_ENC_CHARACTER, dim_char, dim_char])

        # Monster embedding
        self._embedding_monster = MLP([DIM_ENC_MONSTER, dim_monster, dim_monster])

        # Energy embedding
        self._embedding_energy = MLP([DIM_ENC_ENERGY, dim_energy, dim_energy])

        # Transformer
        self._card_transformer_1 = CardTransformer(dim_card, 4, dim_card)
        self._card_transformer_2 = CardTransformer(dim_card, 4, dim_card)

        #
        self._mlp_other = MLP(
            [
                dim_char + dim_monster + dim_energy,
                dim_char + dim_monster + dim_energy,
                dim_char + dim_monster + dim_energy,
            ]
        )

        # Two actions for every card (select/play card, discard card) for now
        self._mlp_card = nn.Sequential(
            MLP(
                [
                    dim_char + dim_monster + dim_energy + 3 + 4 * dim_card,
                    dim_char + dim_monster + dim_energy + 3 + 4 * dim_card,
                    dim_char + dim_monster + dim_energy + 3 + 4 * dim_card,
                ]
            ),
            nn.Linear(
                dim_char + dim_monster + dim_energy + 3 + 4 * dim_card,
                2,
            ),
        )

        # One action for every monster (only one monster for now)
        self._mlp_monster = nn.Sequential(
            MLP(
                [
                    dim_char + dim_monster + dim_energy + 3 + 4 * dim_card,
                    dim_char + dim_monster + dim_energy + 3 + 4 * dim_card,
                    dim_char + dim_monster + dim_energy + 3 + 4 * dim_card,
                ]
            ),
            nn.Linear(
                dim_char + dim_monster + dim_energy + 3 + 4 * dim_card,
                1,
            ),
        )
        # One action for end-of-turn
        self._mlp_end_turn = nn.Sequential(
            MLP(
                [
                    dim_char + dim_monster + dim_energy + 3 + 3 * dim_card,
                    dim_char + dim_monster + dim_energy + 3 + 3 * dim_card,
                    dim_char + dim_monster + dim_energy + 3 + 3 * dim_card,
                ]
            ),
            nn.Linear(
                dim_char + dim_monster + dim_energy + 3 + 3 * dim_card,
                1,
            ),
        )

    def forward(
        self,
        x_mask_hand: torch.Tensor,
        x_mask_draw: torch.Tensor,
        x_mask_disc: torch.Tensor,
        x_card_active_mask: torch.Tensor,
        x_card_hand: torch.Tensor,
        x_card_draw: torch.Tensor,
        x_card_disc: torch.Tensor,
        x_char: torch.Tensor,
        x_monster: torch.Tensor,
        x_energy: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x_mask_hand.shape[0]

        max_size_hand = x_card_hand.shape[1]
        max_size_draw = x_card_draw.shape[1]
        max_size_disc = x_card_disc.shape[1]

        # Encode hand
        x_mask_hand_v = x_mask_hand.view(batch_size, max_size_hand, 1).expand(
            batch_size, max_size_hand, self._dim_card
        )
        x_mask_hand = x_mask_hand.to(torch.bool)
        x_card_hand = self._embedding_card(x_card_hand)
        x_card_hand = self._card_transformer_1(x_card_hand, ~x_mask_hand)
        x_card_hand = self._card_transformer_2(x_card_hand, ~x_mask_hand)
        x_card_hand *= x_mask_hand_v

        # Encode draw
        x_mask_draw_v = x_mask_draw.view(batch_size, max_size_draw, 1).expand(
            batch_size, max_size_draw, self._dim_card
        )
        x_mask_draw = x_mask_draw.to(torch.bool)
        x_card_draw = self._embedding_card(x_card_draw)
        x_card_draw = self._card_transformer_1(x_card_draw, ~x_mask_draw)
        x_card_draw = self._card_transformer_2(x_card_draw, ~x_mask_draw)
        x_card_draw *= x_mask_draw_v

        # Encode discard
        x_mask_disc_v = x_mask_disc.view(batch_size, max_size_disc, 1).expand(
            batch_size, max_size_disc, self._dim_card
        )
        x_mask_disc = x_mask_disc.to(torch.bool)
        x_card_disc = self._embedding_card(x_card_disc)
        x_card_disc = self._card_transformer_1(x_card_disc, ~x_mask_disc)
        x_card_disc = self._card_transformer_2(x_card_disc, ~x_mask_disc)
        x_card_disc *= x_mask_disc_v

        # Other (monster, character, energy)
        x_monster = self._embedding_monster(x_monster)
        x_char = self._embedding_char(x_char)
        x_energy = self._embedding_energy(x_energy)
        x_other = self._mlp_other(
            torch.cat(
                [
                    x_monster,
                    x_char,
                    x_energy,
                ],
                dim=1,
            )
        )
        # Concatenate
        x_len_hand = torch.sum(x_mask_hand, dim=1, keepdim=True, dtype=torch.float32)
        x_len_hand_clamp = torch.clamp(x_len_hand, min=1.0)  # To avoid division by zero in mean
        x_len_draw = torch.sum(x_mask_draw, dim=1, keepdim=True, dtype=torch.float32)
        x_len_draw_clamp = torch.clamp(x_len_draw, min=1.0)  # To avoid division by zero in mean
        x_len_disc = torch.sum(x_mask_disc, dim=1, keepdim=True, dtype=torch.float32)
        x_len_disc_clamp = torch.clamp(x_len_disc, min=1.0)  # To avoid division by zero in mean

        x_global = torch.cat(
            [
                torch.sum(x_card_hand, dim=1) / x_len_hand_clamp,  # Mean
                torch.sum(x_card_draw, dim=1) / x_len_draw_clamp,  # Mean
                torch.sum(x_card_disc, dim=1) / x_len_disc_clamp,  # Mean
                x_len_hand / max_size_hand,
                x_len_draw / max_size_draw,
                x_len_disc / max_size_disc,
                x_other,
            ],
            dim=1,
        )

        # Calculate actions
        x_card = self._mlp_card(
            torch.cat(
                [
                    x_global.view(batch_size, 1, -1).expand(batch_size, max_size_hand, -1),
                    x_card_hand,
                ],
                dim=2,
            )
        )
        x_card_active = torch.sum(
            (
                x_card_active_mask.view(batch_size, max_size_hand, 1).expand(
                    batch_size, max_size_hand, self._dim_card
                )
                * x_card_hand
            ),
            dim=1,
        )
        x_monster = self._mlp_monster(
            torch.cat(
                [
                    x_global,
                    x_card_active,
                ],
                dim=1,
            )
        )
        x_end_turn = self._mlp_end_turn(x_global)

        return torch.cat(
            [
                torch.flatten(x_card.permute(0, 2, 1), start_dim=1),
                x_monster,
                x_end_turn,
            ],
            dim=1,
        )
