"""
Core encoder module that processes game state into embeddings.

The core takes raw game state encodings and produces:
1. Entity embeddings (cards, monsters, character, energy) via transformer
2. Map encoding
3. A global context vector (pooled entity embeddings)
"""

from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.nn as nn

from src.game.const import MAX_MONSTERS
from src.game.const import MAX_SIZE_COMBAT_CARD_REWARD
from src.game.const import MAX_SIZE_DECK
from src.game.const import MAX_SIZE_DISC_PILE
from src.game.const import MAX_SIZE_DRAW_PILE
from src.game.const import MAX_SIZE_HAND
from src.rl.encoding.fsm import get_encoding_dim_fsm
from src.rl.encoding.state import XGameState
from src.rl.models.entity_projector import EntityProjector
from src.rl.models.entity_transformer import EntityTransformer
from src.rl.models.map_encoder import MapEncoder


_FSM_DIM = get_encoding_dim_fsm()


class EntityType(IntEnum):
    """Entity type indices for type embeddings."""

    HAND = 0
    DRAW = 1
    DISC = 2
    DECK = 3
    COMBAT_REWARD = 4
    MONSTER = 5
    CHARACTER = 6
    ENERGY = 7


_NUM_ENTITY_TYPES = len(EntityType)


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
        Mean tensor (B, D)
    """
    # Zero out padded positions
    x_masked = x * torch.unsqueeze(mask, -1)

    # Sum and divide by actual length
    x_sum = torch.sum(x_masked, dim=1)
    x_len = torch.clamp(torch.sum(mask, dim=1, keepdim=True).float(), min=1.0)
    x_mean = x_sum / x_len

    return x_mean


def _calculate_masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Calculate max over sequence dimension, respecting padding mask.

    Args:
        x: Tensor of shape (B, S, D)
        mask: Mask of shape (B, S), True/1.0 = valid

    Returns:
        Max tensor (B, D)
    """
    mask_bool = mask.bool()

    # Set padded positions to -inf so they don't affect max
    mask_expanded = torch.unsqueeze(mask_bool, -1)  # (B, S, 1)
    x_masked = torch.where(mask_expanded, x, torch.full_like(x, float("-inf")))

    # Max over sequence dimension
    x_max, _ = torch.max(x_masked, dim=1)

    # Handle empty sequences: if all positions masked, return zeros instead of -inf
    all_masked = ~torch.any(mask_bool, dim=1, keepdim=True)  # (B, 1)
    x_max = torch.where(all_masked, torch.zeros_like(x_max), x_max)

    return x_max


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
    2. Add learned type embeddings to distinguish entity sources
    3. Pass all entities through transformer for cross-entity attention
    4. Encode map separately
    5. Pool entities per-type (mean + max for each) into global context vector
    """

    def __init__(
        self,
        dim_entity: int,
        dim_global: int,
        transformer_dim_ff: int,
        transformer_num_heads: int,
        transformer_num_blocks: int,
        map_encoder_kernel_size: int,
        map_encoder_dim: int,
    ):
        """
        Args:
            dim_entity: Embedding dimension for all entities
            dim_global: Dimension of the global context vector (output of pooling projection)
            transformer_dim_ff: Feedforward dimension in transformer blocks
            transformer_num_heads: Number of attention heads
            transformer_num_blocks: Number of transformer blocks
            map_encoder_kernel_size: Kernel size for map CNN encoder
            map_encoder_dim: Output dimension of map encoder
        """
        super().__init__()

        self._dim_entity = dim_entity
        self._dim_global = dim_global
        self._map_encoder_dim = map_encoder_dim

        # Entity projector: project each entity type to shared dimension
        self._entity_projector = EntityProjector(dim_entity)

        # Type embeddings: learnable embeddings for each entity type
        # Allows transformer to distinguish hand cards from draw pile cards, etc.
        self._type_embeddings = nn.Embedding(_NUM_ENTITY_TYPES, dim_entity)

        # Entity transformer: cross-entity attention
        self._entity_transformer = EntityTransformer(
            dim_entity,
            transformer_dim_ff,
            transformer_num_heads,
            transformer_num_blocks,
        )

        # Map encoder
        self._map_encoder = MapEncoder(map_encoder_kernel_size, map_encoder_dim)

        # Global context projection with discriminated entity aggregation:
        # - 6 sequence entity types (hand, draw, disc, deck, reward, monsters): mean + max each = 12 * dim
        # - 2 singleton entities (character, energy): 2 * dim
        # - Map encoding: map_encoder_dim
        # - FSM state: FSM_DIM
        _num_seq_entity_types = 6  # hand, draw, disc, deck, reward, monsters
        _num_singleton_entities = 2  # character, energy
        global_input_dim = (
            _num_seq_entity_types * 2 * dim_entity  # mean + max for each sequence type
            + _num_singleton_entities * dim_entity  # character + energy
            + map_encoder_dim
            + _FSM_DIM
        )
        self._global_projection = nn.Sequential(
            nn.Linear(global_input_dim, dim_global),
            nn.ReLU(),
            nn.Linear(dim_global, dim_global),
        )

        # Pre-build type indices (fixed structure, batch-independent)
        # Shape (1, total_entities) — expanded to (B, total_entities) in forward
        _type_indices = torch.cat(
            [
                torch.full((1, MAX_SIZE_HAND), EntityType.HAND),
                torch.full((1, MAX_SIZE_DRAW_PILE), EntityType.DRAW),
                torch.full((1, MAX_SIZE_DISC_PILE), EntityType.DISC),
                torch.full((1, MAX_SIZE_DECK), EntityType.DECK),
                torch.full((1, MAX_SIZE_COMBAT_CARD_REWARD), EntityType.COMBAT_REWARD),
                torch.full((1, MAX_MONSTERS), EntityType.MONSTER),
                torch.full((1, 1), EntityType.CHARACTER),
                torch.full((1, 1), EntityType.ENERGY),
            ],
            dim=1,
        )
        self.register_buffer("_type_indices", _type_indices)

    @property
    def dim_map(self) -> int:
        return self._map_encoder_dim

    @property
    def dim_global(self) -> int:
        return self._dim_global

    def forward(self, x_game_state: XGameState) -> CoreOutput:
        batch_size = x_game_state.x_hand.shape[0]

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
        # Health/block and modifiers projected via shared weights
        x_card_proj, x_monsters_proj, x_character_proj, x_energy_proj = self._entity_projector(
            x_card,
            x_game_state.x_monsters,
            x_game_state.x_monster_health_block,
            x_game_state.x_monster_modifiers,
            x_game_state.x_character,
            x_game_state.x_character_health_block,
            x_game_state.x_character_modifiers,
            x_game_state.x_energy,
        )

        # Expand cached type indices to batch size (no memory allocation)
        type_indices = self._type_indices.expand(batch_size, -1)  # (B, total_entities)

        # Get type embeddings for all positions
        type_emb = self._type_embeddings(type_indices)  # (B, total_entities, dim_entity)

        # Concatenate all entities
        x_entity_cat = torch.cat(
            [
                x_card_proj,
                x_monsters_proj,
                torch.unsqueeze(x_character_proj, 1),
                torch.unsqueeze(x_energy_proj, 1),
            ],
            dim=1,
        )

        # Add type embeddings to entity embeddings
        x_entity_cat = x_entity_cat + type_emb

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
        # Invert mask: encoding uses True=valid, but PyTorch MHA key_padding_mask expects True=padded
        x_entity_mask = x_entity_mask.bool()
        x_entity_mask = ~x_entity_mask
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

        # Create global context with discriminated entity aggregation
        # Compute mean + max for each entity type separately
        x_hand_mean = _calculate_masked_mean(x_hand_out, x_game_state.x_hand_mask_pad)
        x_hand_max = _calculate_masked_max(x_hand_out, x_game_state.x_hand_mask_pad)

        x_draw_mean = _calculate_masked_mean(x_draw_out, x_game_state.x_draw_mask_pad)
        x_draw_max = _calculate_masked_max(x_draw_out, x_game_state.x_draw_mask_pad)

        x_disc_mean = _calculate_masked_mean(x_disc_out, x_game_state.x_disc_mask_pad)
        x_disc_max = _calculate_masked_max(x_disc_out, x_game_state.x_disc_mask_pad)

        x_deck_mean = _calculate_masked_mean(x_deck_out, x_game_state.x_deck_mask_pad)
        x_deck_max = _calculate_masked_max(x_deck_out, x_game_state.x_deck_mask_pad)

        x_reward_mean = _calculate_masked_mean(
            x_combat_reward_out, x_game_state.x_combat_reward_mask_pad
        )
        x_reward_max = _calculate_masked_max(
            x_combat_reward_out, x_game_state.x_combat_reward_mask_pad
        )

        x_monsters_mean = _calculate_masked_mean(x_monsters_out, x_game_state.x_monsters_mask_pad)
        x_monsters_max = _calculate_masked_max(x_monsters_out, x_game_state.x_monsters_mask_pad)

        # Concatenate all aggregated features
        x_global = self._global_projection(
            torch.cat(
                [
                    # Sequence entity aggregations (mean + max)
                    x_hand_mean,
                    x_hand_max,
                    x_draw_mean,
                    x_draw_max,
                    x_disc_mean,
                    x_disc_max,
                    x_deck_mean,
                    x_deck_max,
                    x_reward_mean,
                    x_reward_max,
                    x_monsters_mean,
                    x_monsters_max,
                    # Singleton entities
                    x_character_out,
                    x_energy_out,
                    # Map and FSM
                    x_map,
                    x_game_state.x_fsm,
                ],
                dim=1,
            )
        )

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
