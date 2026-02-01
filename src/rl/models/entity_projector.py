import torch
import torch.nn as nn

from src.rl.encoding.card import get_encoding_dim_card
from src.rl.encoding.character import get_encoding_dim_character
from src.rl.encoding.energy import get_encoding_dim_energy
from src.rl.encoding.monster import get_encoding_dim_monster


_INPUT_DIM_CARD = get_encoding_dim_card()
_INPUT_DIM_MONSTER = get_encoding_dim_monster()
_INPUT_DIM_CHARACTER = get_encoding_dim_character()
_INPUT_DIM_ENERGY = get_encoding_dim_energy()


class EntityProjector(nn.Module):
    def __init__(self, dim_embedding: int):
        super().__init__()

        self._dim_embedding = dim_embedding

        # Card
        self._projection_card = nn.Sequential(
            nn.Linear(_INPUT_DIM_CARD, dim_embedding),
            nn.ReLU(),
            nn.Linear(dim_embedding, dim_embedding),
        )

        # Monsters
        self._projection_monster = nn.Sequential(
            nn.Linear(_INPUT_DIM_MONSTER, dim_embedding),
            nn.ReLU(),
            nn.Linear(dim_embedding, dim_embedding),
        )

        # Character
        self._projection_character = nn.Sequential(
            nn.Linear(_INPUT_DIM_CHARACTER, dim_embedding),
            nn.ReLU(),
            nn.Linear(dim_embedding, dim_embedding),
        )

        # Energy
        self._projection_energy = nn.Sequential(
            nn.Linear(_INPUT_DIM_ENERGY, dim_embedding),
            nn.ReLU(),
            nn.Linear(dim_embedding, dim_embedding),
        )

    def forward(
        self,
        x_card: torch.Tensor,
        x_monsters: torch.Tensor,
        x_character: torch.Tensor,
        x_energy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self._projection_card(x_card),
            self._projection_monster(x_monsters),
            self._projection_character(x_character),
            self._projection_energy(x_energy),
        )
