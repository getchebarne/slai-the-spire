"""
Actor-Critic model with hierarchical action heads.

Architecture:
1. Core encoder processes game state → entity embeddings + global context
2. Primary head selects ActionChoice (unambiguous action + routing)
3. Secondary heads select specific indices (grouped by head type)
4. Value head estimates state value

Key design: ActionChoice directly determines which secondary head to use,
eliminating FSM from the forward pass entirely. FSM is only used when
building masks (before calling forward).
"""

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn

from src.game.action import Action
from src.game.const import MAP_WIDTH
from src.rl.action_space.types import ActionChoice
from src.rl.action_space.types import CHOICE_TO_ACTION_TYPE
from src.rl.action_space.types import CHOICE_TO_HEAD
from src.rl.action_space.types import CHOICE_TO_HEAD_IDX
from src.rl.action_space.types import HEAD_TYPE_NONE
from src.rl.action_space.types import HeadType
from src.rl.action_space.types import NUM_ACTION_CHOICES
from src.rl.encoding.state import XGameState
from src.rl.models.core import Core
from src.rl.models.core import CoreOutput
from src.rl.models.heads import greedy_select_grouped
from src.rl.models.heads import HeadActionType
from src.rl.models.heads import HeadCardDiscard
from src.rl.models.heads import HeadCardPlay
from src.rl.models.heads import HeadCardRewardSelect
from src.rl.models.heads import HeadCardUpgrade
from src.rl.models.heads import HeadMapSelect
from src.rl.models.heads import HeadMonsterSelect
from src.rl.models.heads import HeadValue


# =============================================================================
# Output Types
# =============================================================================


class ForwardOutput(NamedTuple):
    """
    Output from forward pass. All tensors are (B,) or (B, 1).

    This is the primary output type - simple tensors, easy to work with.
    """

    # Primary action choice (index into ActionChoice enum)
    action_choices: torch.Tensor  # (B,) int64
    action_choice_log_probs: torch.Tensor  # (B,)

    # Secondary index (-1 if terminal action)
    secondary_indices: torch.Tensor  # (B,) int64
    secondary_log_probs: torch.Tensor  # (B,)

    # Value estimate
    values: torch.Tensor  # (B, 1)

    def get_action(self, idx: int) -> Action:
        """Convert to game Action for sample at index."""
        choice = ActionChoice(self.action_choices[idx].item())
        action_type = CHOICE_TO_ACTION_TYPE[choice]

        sec_idx = self.secondary_indices[idx].item()
        index = None if sec_idx < 0 else sec_idx

        return Action(type=action_type, index=index)

    def get_log_prob(self, idx: int) -> torch.Tensor:
        """Get total log prob for sample at index."""
        choice = ActionChoice(self.action_choices[idx].item())

        if CHOICE_TO_HEAD[choice] is None:
            # Terminal action - only primary log prob
            return self.action_choice_log_probs[idx]

        return self.action_choice_log_probs[idx] + self.secondary_log_probs[idx]


@dataclass
class SingleOutput:
    """Convenience wrapper for single-sample inference."""

    action_choice: ActionChoice
    action_choice_log_prob: torch.Tensor
    secondary_index: int  # -1 if terminal
    secondary_log_prob: torch.Tensor
    value: torch.Tensor

    def to_action(self) -> Action:
        action_type = CHOICE_TO_ACTION_TYPE[self.action_choice]
        index = None if self.secondary_index < 0 else self.secondary_index
        return Action(type=action_type, index=index)

    @property
    def log_prob(self) -> torch.Tensor:
        if CHOICE_TO_HEAD[self.action_choice] is None:
            return self.action_choice_log_prob
        return self.action_choice_log_prob + self.secondary_log_prob


# =============================================================================
# Helper Functions
# =============================================================================


def _slice_core_output(core_out: CoreOutput, indices: torch.Tensor) -> CoreOutput:
    """Slice CoreOutput to specific batch indices."""
    return CoreOutput(
        x_hand=core_out.x_hand[indices],
        x_draw=core_out.x_draw[indices],
        x_disc=core_out.x_disc[indices],
        x_deck=core_out.x_deck[indices],
        x_combat_reward=core_out.x_combat_reward[indices],
        x_monsters=core_out.x_monsters[indices],
        x_character=core_out.x_character[indices],
        x_energy=core_out.x_energy[indices],
        x_entity=core_out.x_entity[indices],
        x_entity_mask=core_out.x_entity_mask[indices],
        x_map=core_out.x_map[indices],
        x_global=core_out.x_global[indices],
    )


# =============================================================================
# Model
# =============================================================================


class ActorCritic(nn.Module):
    """
    Actor-Critic with clean batched forward pass.

    Forward pass efficiency:
    - Core encoder: 1 call (all samples)
    - Primary head: 1 call (all samples)
    - Secondary heads: ≤6 calls (only heads with samples needing them)
    - Value head: 1 call (all samples)

    No FSM in forward - ActionChoice directly determines routing.
    """

    def __init__(
        self,
        dim_entity: int = 128,
        transformer_dim_ff: int = 256,
        transformer_num_heads: int = 4,
        transformer_num_blocks: int = 2,
        map_encoder_kernel_size: int = 3,
        map_encoder_dim: int = 32,
        dim_ff_primary: int = 128,
        dim_ff_card: int = 128,
        dim_ff_monster: int = 128,
        dim_ff_map: int = 128,
        dim_ff_value: int = 128,
    ):
        super().__init__()

        # Core encoder
        self.core = Core(
            dim_entity=dim_entity,
            transformer_dim_ff=transformer_dim_ff,
            transformer_num_heads=transformer_num_heads,
            transformer_num_blocks=transformer_num_blocks,
            map_encoder_kernel_size=map_encoder_kernel_size,
            map_encoder_dim=map_encoder_dim,
        )

        dim_global = self.core.dim_global
        dim_map = self.core.dim_map

        # Primary head: outputs ActionChoice
        self.head_primary = HeadActionType(
            dim_global=dim_global,
            dim_ff=dim_ff_primary,
            max_action_types=NUM_ACTION_CHOICES,
        )

        # Secondary heads
        self.head_card_play = HeadCardPlay(dim_entity, dim_global, dim_ff_card)
        self.head_card_discard = HeadCardDiscard(dim_entity, dim_global, dim_ff_card)
        self.head_card_reward = HeadCardRewardSelect(dim_entity, dim_global, dim_ff_card)
        self.head_card_upgrade = HeadCardUpgrade(dim_entity, dim_global, dim_ff_card)
        self.head_monster_select = HeadMonsterSelect(dim_entity, dim_global, dim_ff_monster)
        self.head_map_select = HeadMapSelect(dim_map, dim_global, dim_ff_map, MAP_WIDTH)

        # Value head
        self.head_value = HeadValue(dim_global, dim_ff_value)

        # Head registry
        self._heads: dict[HeadType, nn.Module] = {
            HeadType.CARD_PLAY: self.head_card_play,
            HeadType.CARD_DISCARD: self.head_card_discard,
            HeadType.CARD_REWARD_SELECT: self.head_card_reward,
            HeadType.CARD_UPGRADE: self.head_card_upgrade,
            HeadType.MONSTER_SELECT: self.head_monster_select,
            HeadType.MAP_SELECT: self.head_map_select,
        }

    def _get_entities(self, head_type: HeadType, core_out: CoreOutput) -> torch.Tensor:
        """Get entity tensor for a secondary head."""
        match head_type:
            case HeadType.CARD_PLAY | HeadType.CARD_DISCARD:
                return core_out.x_hand
            case HeadType.CARD_REWARD_SELECT:
                return core_out.x_combat_reward
            case HeadType.CARD_UPGRADE:
                return core_out.x_deck
            case HeadType.MONSTER_SELECT:
                return core_out.x_monsters
            case _:
                raise ValueError(f"Unknown head type: {head_type}")

    def _run_secondary(
        self,
        head_type: HeadType,
        core_out: CoreOutput,
        mask: torch.Tensor,
        sample: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a secondary head.

        Returns (indices, log_probs) tensors.
        """
        if head_type == HeadType.MAP_SELECT:
            out = self.head_map_select(core_out.x_map, core_out.x_global, mask, sample)
        else:
            head = self._heads[head_type]
            entities = self._get_entities(head_type, core_out)
            out = head(entities, core_out.x_global, mask, sample)

        if sample:
            return out.indices, out.log_probs
        else:
            return greedy_select_grouped(out.logits), torch.zeros(
                out.logits.shape[0], device=out.logits.device
            )

    def forward(
        self,
        x_game_state: XGameState,
        primary_mask: torch.Tensor,
        secondary_masks: dict[HeadType, torch.Tensor],
        sample: bool = True,
    ) -> ForwardOutput:
        """
        Batched forward pass.

        Args:
            x_game_state: Encoded game state
            primary_mask: Valid ActionChoice mask (B, NUM_ACTION_CHOICES)
            secondary_masks: Per-head masks {HeadType: (B, head_output_size)}
            sample: Whether to sample (True) or argmax (False)

        Returns:
            ForwardOutput with per-sample tensors
        """
        device = x_game_state.x_hand.device

        # =================================================================
        # 1. Core encoder (all samples)
        # =================================================================
        core_out = self.core(x_game_state)
        B = core_out.x_global.shape[0]

        # =================================================================
        # 2. Value head (all samples)
        # =================================================================
        values = self.head_value(core_out.x_global)

        # =================================================================
        # 3. Primary head (all samples) → ActionChoice
        # =================================================================
        primary_out = self.head_primary(core_out.x_global, primary_mask, sample)

        if sample:
            action_choices = primary_out.indices
            action_choice_log_probs = primary_out.log_probs
        else:
            action_choices = torch.argmax(primary_out.logits, dim=-1)
            action_choice_log_probs = torch.zeros(B, device=device)

        # =================================================================
        # 4. Get head types for all samples (vectorized, no .item()!)
        # =================================================================
        # Move lookup tensor to device if needed (cached after first call)
        head_type_lookup = CHOICE_TO_HEAD_IDX.to(device)
        head_type_indices = head_type_lookup[action_choices]  # (B,) tensor

        # =================================================================
        # 5. Run secondary heads (grouped by head type)
        # =================================================================
        secondary_indices = torch.full((B,), -1, dtype=torch.long, device=device)
        secondary_log_probs = torch.zeros(B, device=device)

        for head_type in HeadType:
            # Find samples needing this head (vectorized comparison)
            sample_mask = head_type_indices == head_type
            if not torch.any(sample_mask):
                continue

            # Get indices of matching samples
            idx = torch.nonzero(sample_mask, as_tuple=True)[0]

            # Slice for this head's samples
            subset_core = _slice_core_output(core_out, idx)
            subset_mask = secondary_masks[head_type][idx]

            # Run head
            indices, log_probs = self._run_secondary(head_type, subset_core, subset_mask, sample)

            # Scatter results back
            secondary_indices[idx] = indices
            secondary_log_probs[idx] = log_probs

        return ForwardOutput(
            action_choices=action_choices,
            action_choice_log_probs=action_choice_log_probs,
            secondary_indices=secondary_indices,
            secondary_log_probs=secondary_log_probs,
            values=values,
        )

    def forward_single(
        self,
        x_game_state: XGameState,
        primary_mask: torch.Tensor,
        secondary_masks: dict[HeadType, torch.Tensor],
        sample: bool = True,
    ) -> SingleOutput:
        """Convenience method for single sample."""
        out = self.forward(x_game_state, primary_mask, secondary_masks, sample)

        return SingleOutput(
            action_choice=ActionChoice(out.action_choices[0].item()),
            action_choice_log_prob=out.action_choice_log_probs[0],
            secondary_index=out.secondary_indices[0].item(),
            secondary_log_prob=out.secondary_log_probs[0],
            value=out.values[0],
        )
