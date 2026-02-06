import torch
import torch.nn as nn

from src.rl.encoding.actor import get_encoding_dim_actor_modifiers
from src.rl.encoding.card import get_encoding_dim_card
from src.rl.encoding.character import get_encoding_dim_character
from src.rl.encoding.energy import get_encoding_dim_energy
from src.rl.encoding.health_block import get_encoding_dim_health_block
from src.rl.encoding.monster import get_encoding_dim_monster


_INPUT_DIM_CARD = get_encoding_dim_card()
_INPUT_DIM_MONSTER = get_encoding_dim_monster()
_INPUT_DIM_CHARACTER = get_encoding_dim_character()
_INPUT_DIM_ENERGY = get_encoding_dim_energy()
_INPUT_DIM_HEALTH_BLOCK = get_encoding_dim_health_block()
_INPUT_DIM_MODIFIERS = get_encoding_dim_actor_modifiers()


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

        # Monsters (identity + intent, excludes health/block and modifiers)
        self._projection_monster = nn.Sequential(
            nn.Linear(_INPUT_DIM_MONSTER, dim_embedding),
            nn.ReLU(),
            nn.Linear(dim_embedding, dim_embedding),
        )

        # Character (survivability, excludes health/block and modifiers)
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

        # Shared health/block projector (used by both character and monsters)
        self._projection_health_block = nn.Sequential(
            nn.Linear(_INPUT_DIM_HEALTH_BLOCK, dim_embedding),
            nn.ReLU(),
            nn.Linear(dim_embedding, dim_embedding),
        )

        # Shared modifier projector (used by both character and monsters)
        self._projection_modifier = nn.Sequential(
            nn.Linear(_INPUT_DIM_MODIFIERS, dim_embedding),
            nn.ReLU(),
            nn.Linear(dim_embedding, dim_embedding),
        )

        # Single LayerNorm applied after full additive composition
        self._layer_norm = nn.LayerNorm(dim_embedding)

    def forward(
        self,
        x_card: torch.Tensor,
        x_monsters: torch.Tensor,
        x_monster_health_block: torch.Tensor,
        x_monster_modifiers: torch.Tensor,
        x_character: torch.Tensor,
        x_character_health_block: torch.Tensor,
        x_character_modifiers: torch.Tensor,
        x_energy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Entity-specific projections
        x_card_proj = self._projection_card(x_card)
        x_monster_proj = self._projection_monster(x_monsters)
        x_character_proj = self._projection_character(x_character)
        x_energy_proj = self._projection_energy(x_energy)

        # Shared health/block projection (additive composition)
        x_monster_proj = x_monster_proj + self._projection_health_block(x_monster_health_block)
        x_character_proj = x_character_proj + self._projection_health_block(
            x_character_health_block
        )

        # Shared modifier projection (additive composition)
        x_monster_proj = x_monster_proj + self._projection_modifier(x_monster_modifiers)
        x_character_proj = x_character_proj + self._projection_modifier(x_character_modifiers)

        # Normalize all entity embeddings after full composition
        x_card_proj = self._layer_norm(x_card_proj)
        x_monster_proj = self._layer_norm(x_monster_proj)
        x_character_proj = self._layer_norm(x_character_proj)
        x_energy_proj = self._layer_norm(x_energy_proj)

        return x_card_proj, x_monster_proj, x_character_proj, x_energy_proj
