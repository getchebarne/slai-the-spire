"""
Core encoder module that processes game state into embeddings.

The core takes raw game state encodings and produces:
1. Entity embeddings (cards, monsters, character, energy) via transformer
2. Map encoding
3. A global context vector (pooled entity embeddings)
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.game.const import MAX_MONSTERS
from src.game.const import MAX_SIZE_COMBAT_CARD_REWARD
from src.game.const import MAX_SIZE_DECK
from src.game.const import MAX_SIZE_DISC_PILE
from src.game.const import MAX_SIZE_DRAW_PILE
from src.game.const import MAX_SIZE_HAND
from src.rl.encoding.state import XGameState
from src.rl.models.entity_projector import EntityProjector
from src.rl.models.entity_transformer import EntityTransformer
from src.rl.models.map_encoder import MapEncoder


@dataclass
class CoreOutput:
    """Output from the core encoder."""

    # Individual entity embeddings (after transformer)
    x_hand: torch.Tensor  # (B, MAX_SIZE_HAND, dim_entity)
    x_draw: torch.Tensor  # (B, MAX_SIZE_DRAW_PILE, dim_entity)
    x_disc: torch.Tensor  # (B, MAX_SIZE_DISC_PILE, dim_entity)
    x_deck: torch.Tensor  # (B, MAX_SIZE_DECK, dim_entity)
    x_combat_reward: torch.Tensor  # (B, MAX_SIZE_COMBAT_CARD_REWARD, dim_entity)
    x_monsters: torch.Tensor  # (B, MAX_MONSTERS, dim_entity)
    x_character: torch.Tensor  # (B, dim_entity)
    x_energy: torch.Tensor  # (B, dim_entity)

    # Concatenated entity tensor (for attention-based pooling if needed)
    x_entity: torch.Tensor  # (B, total_entities, dim_entity)
    x_entity_mask: torch.Tensor  # (B, total_entities)

    # Map encoding
    x_map: torch.Tensor  # (B, dim_map)

    # Global context (pooled entities + map)
    x_global: torch.Tensor  # (B, dim_global)


def _calculate_masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Calculate mean over sequence dimension, respecting padding mask.

    Args:
        x: Tensor of shape (B, S, D)
        mask: Boolean mask of shape (B, S), True = valid

    Returns:
        Tuple of (mean tensor (B, D), sequence lengths (B, 1))
    """
    # Zero out padded positions
    x_masked = x * torch.unsqueeze(mask, -1)

    # Sum and divide by actual length
    x_sum = torch.sum(x_masked, dim=1)
    x_len = torch.clamp(torch.sum(mask, dim=1, keepdim=True).float(), min=1.0)
    x_mean = x_sum / x_len

    return x_mean


def _undo_entity_concatenation(
    x_entity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split concatenated entity tensor back into components."""
    total_card_len = (
        MAX_SIZE_HAND
        + MAX_SIZE_DRAW_PILE
        + MAX_SIZE_DISC_PILE
        + MAX_SIZE_DECK
        + MAX_SIZE_COMBAT_CARD_REWARD
    )

    idx = 0

    # Cards
    x_card = x_entity[:, idx : idx + total_card_len, :]
    idx += total_card_len

    # Monsters
    x_monsters = x_entity[:, idx : idx + MAX_MONSTERS, :]
    idx += MAX_MONSTERS

    # Character (single entity)
    x_character = x_entity[:, idx, :]
    idx += 1

    # Energy (single entity)
    x_energy = x_entity[:, idx, :]

    return x_card, x_monsters, x_character, x_energy


def _undo_card_concatenation(
    x_card: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split concatenated card tensor back into piles."""
    idx = 0

    x_hand = x_card[:, idx : idx + MAX_SIZE_HAND, :]
    idx += MAX_SIZE_HAND

    x_draw = x_card[:, idx : idx + MAX_SIZE_DRAW_PILE, :]
    idx += MAX_SIZE_DRAW_PILE

    x_disc = x_card[:, idx : idx + MAX_SIZE_DISC_PILE, :]
    idx += MAX_SIZE_DISC_PILE

    x_deck = x_card[:, idx : idx + MAX_SIZE_DECK, :]
    idx += MAX_SIZE_DECK

    x_combat_reward = x_card[:, idx : idx + MAX_SIZE_COMBAT_CARD_REWARD, :]

    return x_hand, x_draw, x_disc, x_deck, x_combat_reward


class Core(nn.Module):
    """
    Core encoder that processes game state into embeddings.

    Architecture:
    1. Project each entity type (cards, monsters, character, energy) to shared dimension
    2. Pass all entities through transformer for cross-entity attention
    3. Encode map separately
    4. Pool entities into global context vector
    """

    def __init__(
        self,
        dim_entity: int,
        transformer_dim_ff: int,
        transformer_num_heads: int,
        transformer_num_blocks: int,
        map_encoder_kernel_size: int,
        map_encoder_dim: int,
    ):
        """
        Args:
            dim_entity: Embedding dimension for all entities
            transformer_dim_ff: Feedforward dimension in transformer blocks
            transformer_num_heads: Number of attention heads
            transformer_num_blocks: Number of transformer blocks
            map_encoder_kernel_size: Kernel size for map CNN encoder
            map_encoder_dim: Output dimension of map encoder
        """
        super().__init__()

        self._dim_entity = dim_entity
        self._map_encoder_dim = map_encoder_dim

        # Entity projector: project each entity type to shared dimension
        self._entity_projector = EntityProjector(dim_entity)

        # Entity transformer: cross-entity attention
        self._entity_transformer = EntityTransformer(
            dim_entity,
            transformer_dim_ff,
            transformer_num_heads,
            transformer_num_blocks,
        )

        # Map encoder
        self._map_encoder = MapEncoder(map_encoder_kernel_size, map_encoder_dim)

        # Global context projection (entity mean + map → global)
        self._global_projection = nn.Sequential(
            nn.Linear(dim_entity + map_encoder_dim, dim_entity),
            nn.ReLU(),
            nn.Linear(dim_entity, dim_entity),
        )

    @property
    def dim_map(self) -> int:
        return self._map_encoder_dim

    @property
    def dim_global(self) -> int:
        return self._dim_entity

    def forward(self, x_game_state: XGameState) -> CoreOutput:
        # Concatenate all cards (and their masks)
        x_card = torch.cat(
            [
                x_game_state.x_hand,
                x_game_state.x_draw,
                x_game_state.x_disc,
                x_game_state.x_deck,
                x_game_state.x_combat_reward,
            ],
            dim=1,
        )
        # Project entities to shared dimension
        x_card_proj, x_monsters_proj, x_character_proj, x_energy_proj = self._entity_projector(
            x_card,
            x_game_state.x_monsters,
            x_game_state.x_character,
            x_game_state.x_energy,
        )

        # Concatenate all entities and masks for transformer
        x_entity_cat = torch.cat(
            [
                x_card_proj,
                x_monsters_proj,
                x_character_proj.unsqueeze(1),
                x_energy_proj.unsqueeze(1),
            ],
            dim=1,
        )
        x_entity_mask = torch.cat(
            [
                x_game_state.x_hand_mask_pad,
                x_game_state.x_draw_mask_pad,
                x_game_state.x_disc_mask_pad,
                x_game_state.x_deck_mask_pad,
                x_game_state.x_combat_reward_mask_pad,
                x_game_state.x_monsters_mask_pad,
                x_game_state.x_character_mask_pad,
                x_game_state.x_energy_mask_pad,
            ],
            dim=1,
        )

        # Pass through entity transformer
        x_entity = self._entity_transformer(x_entity_cat, x_entity_mask)

        # Undo concatenations to get individual tensors
        x_card_out, x_monsters_out, x_character_out, x_energy_out = _undo_entity_concatenation(
            x_entity
        )
        (
            x_hand_out,
            x_draw_out,
            x_disc_out,
            x_deck_out,
            x_combat_reward_out,
        ) = _undo_card_concatenation(x_card_out)

        # Encode map
        x_map = self._map_encoder(x_game_state.x_map)

        # Create global context: mean pool entities & map
        x_entity_mean = _calculate_masked_mean(x_entity, x_entity_mask)
        x_global = self._global_projection(torch.cat([x_entity_mean, x_map], dim=1))

        return CoreOutput(
            x_hand=x_hand_out,
            x_draw=x_draw_out,
            x_disc=x_disc_out,
            x_deck=x_deck_out,
            x_combat_reward=x_combat_reward_out,
            x_monsters=x_monsters_out,
            x_character=x_character_out,
            x_energy=x_energy_out,
            x_entity=x_entity,
            x_entity_mask=x_entity_mask,
            x_map=x_map,
            x_global=x_global,
        )
