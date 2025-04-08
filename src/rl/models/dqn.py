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


# TODO: parametrize dimensions better, so they are more readable
class DeepQNetwork(nn.Module):
    def __init__(self, dim_card: int):
        super().__init__()

        self._dim_card = dim_card

        # Card embedding
        self._embedding_card = nn.Linear(DIM_ENC_CARD, dim_card)

        # Character embedding
        self._embedding_char = MLP([DIM_ENC_CHARACTER, DIM_ENC_CHARACTER, DIM_ENC_CHARACTER])

        # Monster embedding
        self._embedding_monster = MLP([DIM_ENC_MONSTER, DIM_ENC_MONSTER, DIM_ENC_MONSTER])

        # Energy embedding
        self._embedding_energy = MLP([DIM_ENC_ENERGY, DIM_ENC_ENERGY, DIM_ENC_ENERGY])

        # Other
        self._mlp_other = MLP([DIM_ENC_OTHER, DIM_ENC_OTHER, DIM_ENC_OTHER])

        # Layer normalization after concatenating other and cards
        self._ln_global = nn.LayerNorm(DIM_ENC_OTHER + 3 * dim_card)

        # Two actions for every card (select/play card, discard card) for now
        aux = DIM_ENC_OTHER + 4 * dim_card
        self._mlp_card = nn.Sequential(
            MLP([aux, aux, aux]),
            nn.Linear(aux, 2),
        )

        # One action for every monster (only one monster for now) + 1 for end-of-turn
        self._mlp_monster = nn.Sequential(
            MLP([aux, aux, aux]),
            nn.Linear(aux, 1),
        )
        self._mlp_end_turn = nn.Sequential(
            MLP([aux - dim_card, aux - dim_card, aux - dim_card]),
            nn.Linear(aux - dim_card, 1),
        )

    def forward(
        self,
        x_len_hand: torch.Tensor,
        x_len_draw: torch.Tensor,
        x_len_disc: torch.Tensor,
        x_card_active_mask: torch.Tensor,
        x_card_hand: torch.Tensor,
        x_card_draw: torch.Tensor,
        x_card_disc: torch.Tensor,
        x_char: torch.Tensor,
        x_monster: torch.Tensor,
        x_energy: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x_len_hand.shape[0]

        max_size_hand = x_card_hand.shape[1]
        max_size_draw = x_card_draw.shape[1]
        max_size_disc = x_card_disc.shape[1]

        # Encode hand
        x_card_hand = self._embedding_card(x_card_hand)
        mask_hand = torch.arange(max_size_hand).view(1, max_size_hand).expand(
            batch_size, max_size_hand
        ) < x_len_hand.expand(batch_size, max_size_hand)
        mask_hand = (
            mask_hand.view(batch_size, max_size_hand, 1)
            .expand(batch_size, max_size_hand, self._dim_card)
            .to(torch.float32)
        )
        x_card_hand *= mask_hand

        # Encode draw
        x_card_draw = self._embedding_card(x_card_draw)
        mask_draw = torch.arange(max_size_draw).view(1, max_size_draw).expand(
            batch_size, max_size_draw
        ) < x_len_draw.expand(batch_size, max_size_draw)
        mask_draw = (
            mask_draw.view(batch_size, max_size_draw, 1)
            .expand(batch_size, max_size_draw, self._dim_card)
            .to(torch.float32)
        )
        x_card_draw *= mask_draw

        # Encode disc
        x_card_disc = self._embedding_card(x_card_disc)
        mask_disc = torch.arange(max_size_disc).view(1, max_size_disc).expand(
            batch_size, max_size_disc
        ) < x_len_disc.expand(batch_size, max_size_disc)
        mask_disc = (
            mask_disc.view(batch_size, max_size_disc, 1)
            .expand(batch_size, max_size_disc, self._dim_card)
            .to(torch.float32)
        )
        x_card_disc *= mask_disc

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
        x_global = self._ln_global(
            torch.cat(
                [
                    torch.sum(x_card_hand, dim=1),
                    torch.sum(x_card_draw, dim=1),
                    torch.sum(x_card_disc, dim=1),
                    x_other,
                ],
                dim=1,
            )
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
