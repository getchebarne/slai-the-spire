import torch
import torch.nn as nn

from src.agents.a2c.encode import Encoding
from src.agents.a2c.encode import get_card_encoding_dim
from src.agents.a2c.encode import get_character_encoding_dim
from src.agents.a2c.encode import get_energy_encoding_dim
from src.agents.a2c.encode import get_monster_encoding_dim
from src.agents.a2c.models.mlp import MLP
from src.game.combat.constant import MAX_HAND_SIZE


DIM_ENC_CARD = get_card_encoding_dim()
DIM_ENC_CHARACTER = get_character_encoding_dim()[0]
DIM_ENC_ENERGY = get_energy_encoding_dim()[0]
DIM_ENC_MONSTER = get_monster_encoding_dim()[0]
DIM_OTHER = DIM_ENC_CHARACTER + DIM_ENC_MONSTER + DIM_ENC_ENERGY


class Critic(nn.Module):
    def __init__(self, dim_card: int):
        super().__init__()

        self._dim_card = dim_card

        # Card embedding and transformer
        self._embedding_card_pad = nn.Parameter(torch.randn(DIM_ENC_CARD))
        self._embedding_card = nn.Linear(DIM_ENC_CARD, dim_card)
        self._embedding_card_active = nn.Parameter(torch.randn(dim_card))
        self._embedding_card_discard = nn.Parameter(torch.randn(dim_card))

        # Other state variables (i.e., character, monsters, energy)
        self._embedding_other = MLP([DIM_OTHER, 3 * dim_card, 3 * dim_card])

        # Critic
        self._mlp_value = nn.Sequential(
            MLP([10 * dim_card, 10 * dim_card, 10 * dim_card]),
            nn.Linear(10 * dim_card, 1),
        )

    def forward(self, encoding: Encoding) -> tuple[torch.Tensor, torch.Tensor]:
        # Hand
        x_card_hand = torch.cat(
            [
                encoding.hand,
                self._embedding_card_pad.view((1, DIM_ENC_CARD)).expand(
                    MAX_HAND_SIZE - len(encoding.hand), DIM_ENC_CARD
                ),
            ]
        )
        x_card_hand = self._embedding_card(x_card_hand)

        if encoding.idx_card_active is not None:
            x_card_hand[encoding.idx_card_active] += self._embedding_card_active

        if encoding.effect:
            x_card_hand += self._embedding_card_discard.view((1, self._dim_card)).expand(
                (MAX_HAND_SIZE, self._dim_card)
            )

        # Other
        x_other = self._embedding_other(
            torch.cat(
                [
                    encoding.character,
                    encoding.monster,
                    encoding.energy,
                ]
            )
        )

        # Concatenate
        x_global = torch.cat(
            [
                torch.flatten(x_card_hand),
                x_other,
            ]
        )

        # Calculate value
        x_value = self._mlp_value(x_global)

        return x_value
