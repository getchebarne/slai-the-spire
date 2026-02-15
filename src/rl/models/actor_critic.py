"""
Actor-Critic model with per-FSM-state hierarchical action heads.

Architecture:
1. Core encoder processes game state → entity embeddings + global context
2. Samples are routed by HeadTypePrimary (determined by FSM state)
3. Decision primaries (COMBAT_DEFAULT, CARD_REWARD, REST_SITE):
   - Binary head makes a choice (e.g., end_turn vs play_card)
   - If the "select" choice is picked, a secondary head picks the entity
4. Direct primaries (COMBAT_CARD_DISCARD, COMBAT_MONSTER_SELECT, MAP_SELECT):
   - Selection head directly picks the entity (no binary decision)
5. Value head estimates state value
"""

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn

from src.game.action import Action
from src.game.const import MAP_WIDTH
from src.rl.action_space.masks import MaskBatch
from src.rl.action_space.types import DECISION_PRIMARIES
from src.rl.action_space.types import HeadTypePrimary
from src.rl.action_space.types import HeadTypeSecondary
from src.rl.action_space.types import PRIMARY_TO_SECONDARY
from src.rl.action_space.types import to_action
from src.rl.encoding.state import XGameState
from src.rl.models.core import Core
from src.rl.models.core import CoreOutput
from src.rl.models.heads import HeadBinaryChoice
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
    Output from batched forward pass. All tensors are (B,) or (B, 1).
    """

    # Which primary group each sample belongs to
    head_type_primaries: torch.Tensor  # (B,) int64

    # Primary decision (decision primaries only, -1 for direct primaries)
    primary_indices: torch.Tensor  # (B,) int64
    primary_log_probs: torch.Tensor  # (B,)

    # Entity selection (-1 if terminal, e.g. end_turn/skip/rest)
    selection_indices: torch.Tensor  # (B,) int64
    selection_log_probs: torch.Tensor  # (B,)

    # Value estimate
    values: torch.Tensor  # (B, 1)

    def get_action(self, idx: int) -> Action:
        """Convert to game Action for sample at index."""
        htp = HeadTypePrimary(self.head_type_primaries[idx].item())
        pi = self.primary_indices[idx].item()
        si = self.selection_indices[idx].item()
        return to_action(htp, pi, si)

    def get_log_prob(self, idx: int) -> torch.Tensor:
        """Get total log prob for sample at index."""
        return self.primary_log_probs[idx] + self.selection_log_probs[idx]


@dataclass
class SingleOutput:
    """Convenience wrapper for single-sample inference."""

    head_type_primary: HeadTypePrimary
    primary_index: int  # -1 for direct primaries
    primary_log_prob: torch.Tensor
    selection_index: int  # -1 if terminal
    selection_log_prob: torch.Tensor
    value: torch.Tensor

    def to_action(self) -> Action:
        return to_action(self.head_type_primary, self.primary_index, self.selection_index)

    @property
    def log_prob(self) -> torch.Tensor:
        return self.primary_log_prob + self.selection_log_prob


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

    def __init__(
        self,
        dim_entity: int = 128,
        dim_global: int = 256,
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
            dim_global=dim_global,
            transformer_dim_ff=transformer_dim_ff,
            transformer_num_heads=transformer_num_heads,
            transformer_num_blocks=transformer_num_blocks,
            map_encoder_kernel_size=map_encoder_kernel_size,
            map_encoder_dim=map_encoder_dim,
        )

        dim_global = self.core.dim_global
        dim_map = self.core.dim_map

        # Decision primary heads (binary choice)
        self.head_combat_default = HeadBinaryChoice(dim_global, dim_ff_primary)
        self.head_card_reward = HeadBinaryChoice(dim_global, dim_ff_primary)
        self.head_rest_site = HeadBinaryChoice(dim_global, dim_ff_primary)

        # Entity selection heads
        self.head_card_play = HeadCardPlay(dim_entity, dim_global, dim_ff_card)
        self.head_card_discard = HeadCardDiscard(dim_entity, dim_global, dim_ff_card)
        self.head_card_reward_select = HeadCardRewardSelect(dim_entity, dim_global, dim_ff_card)
        self.head_card_upgrade = HeadCardUpgrade(dim_entity, dim_global, dim_ff_card)
        self.head_monster_select = HeadMonsterSelect(dim_entity, dim_global, dim_ff_monster)
        self.head_map_select = HeadMapSelect(dim_map, dim_global, dim_ff_map, MAP_WIDTH)

        # Value head
        self.head_value = HeadValue(dim_global, dim_ff_value)

        # ---- Registries (for programmatic access) ----

        # Decision primary → binary head
        self._decision_heads: dict[HeadTypePrimary, HeadBinaryChoice] = {
            HeadTypePrimary.COMBAT_DEFAULT: self.head_combat_default,
            HeadTypePrimary.CARD_REWARD: self.head_card_reward,
            HeadTypePrimary.REST_SITE: self.head_rest_site,
        }

        # HeadTypePrimary → entity selection head
        # (MAP_SELECT handled separately due to different input signature)
        self._selection_heads: dict[HeadTypePrimary, nn.Module] = {
            HeadTypePrimary.COMBAT_DEFAULT: self.head_card_play,
            HeadTypePrimary.CARD_REWARD: self.head_card_reward_select,
            HeadTypePrimary.REST_SITE: self.head_card_upgrade,
            HeadTypePrimary.COMBAT_CARD_DISCARD: self.head_card_discard,
            HeadTypePrimary.COMBAT_MONSTER_SELECT: self.head_monster_select,
        }

    def _get_selection_entities(
        self, head_type: HeadTypePrimary, core_out: CoreOutput
    ) -> torch.Tensor:
        """Get the entity tensor for a selection head."""
        match head_type:
            case HeadTypePrimary.COMBAT_DEFAULT | HeadTypePrimary.COMBAT_CARD_DISCARD:
                return core_out.x_hand
            case HeadTypePrimary.CARD_REWARD:
                return core_out.x_combat_reward
            case HeadTypePrimary.REST_SITE:
                return core_out.x_deck
            case HeadTypePrimary.COMBAT_MONSTER_SELECT:
                return core_out.x_monsters
            case _:
                raise ValueError(f"No entity tensor for: {head_type}")

    def _run_selection(
        self,
        head_type: HeadTypePrimary,
        core_out: CoreOutput,
        mask: torch.Tensor,
        sample: bool,
    ):
        """Run the appropriate selection head for a primary type."""
        if head_type == HeadTypePrimary.MAP_SELECT:
            return self.head_map_select(core_out.x_map, core_out.x_global, mask, sample)

        head = self._selection_heads[head_type]
        entities = self._get_selection_entities(head_type, core_out)
        return head(entities, core_out.x_global, mask, sample)

    def forward(
        self,
        x_game_state: XGameState,
        mask_batch: MaskBatch,
        sample: bool = True,
    ) -> ForwardOutput:
        """
        Batched forward pass with per-primary-type routing.

        Args:
            x_game_state: Encoded game state
            mask_batch: Per-HeadTypePrimary routing and masks
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
        # 3. Initialize output tensors
        # =================================================================
        head_type_primaries = torch.full((B,), -1, dtype=torch.long, device=device)
        primary_indices = torch.full((B,), -1, dtype=torch.long, device=device)
        primary_log_probs = torch.zeros(B, device=device)
        selection_indices = torch.full((B,), -1, dtype=torch.long, device=device)
        selection_log_probs = torch.zeros(B, device=device)

        # =================================================================
        # 4. Process each primary group
        # =================================================================
        for head_type in HeadTypePrimary:
            idx = mask_batch.route[head_type]
            if len(idx) == 0:
                continue

            head_type_primaries[idx] = head_type
            subset_core = _slice_core_output(core_out, idx)

            if head_type in DECISION_PRIMARIES:
                # --- Decision primary: binary head + conditional secondary ---
                primary_mask = mask_batch.primary_masks[head_type]
                decision_head = self._decision_heads[head_type]
                out = decision_head(subset_core.x_global, primary_mask, sample)

                if sample:
                    chosen = out.indices
                    primary_indices[idx] = chosen
                    primary_log_probs[idx] = out.log_probs
                else:
                    chosen = torch.argmax(out.logits, dim=-1)
                    primary_indices[idx] = chosen

                # Run secondary head for samples that chose "select" (index == 1)
                needs_secondary = chosen == 1
                if torch.any(needs_secondary):
                    sec_local = torch.nonzero(needs_secondary, as_tuple=True)[0]
                    sec_batch = idx[sec_local]

                    sec_core = _slice_core_output(core_out, sec_batch)
                    sec_mask = mask_batch.selection_masks[head_type][sec_local]

                    sec_out = self._run_selection(head_type, sec_core, sec_mask, sample)

                    if sample:
                        selection_indices[sec_batch] = sec_out.indices
                        selection_log_probs[sec_batch] = sec_out.log_probs
                    else:
                        selection_indices[sec_batch] = torch.argmax(sec_out.logits, dim=-1)

            else:
                # --- Direct primary: selection head only ---
                sel_mask = mask_batch.selection_masks[head_type]
                sel_out = self._run_selection(head_type, subset_core, sel_mask, sample)

                if sample:
                    selection_indices[idx] = sel_out.indices
                    selection_log_probs[idx] = sel_out.log_probs
                else:
                    selection_indices[idx] = torch.argmax(sel_out.logits, dim=-1)

        return ForwardOutput(
            head_type_primaries=head_type_primaries,
            primary_indices=primary_indices,
            primary_log_probs=primary_log_probs,
            selection_indices=selection_indices,
            selection_log_probs=selection_log_probs,
            values=values,
        )

    def forward_single(
        self,
        x_game_state: XGameState,
        mask_batch: MaskBatch,
        sample: bool = True,
    ) -> SingleOutput:
        """Convenience method for single sample."""
        out = self.forward(x_game_state, mask_batch, sample)

        return SingleOutput(
            head_type_primary=HeadTypePrimary(out.head_type_primaries[0].item()),
            primary_index=out.primary_indices[0].item(),
            primary_log_prob=out.primary_log_probs[0],
            selection_index=out.selection_indices[0].item(),
            selection_log_prob=out.selection_log_probs[0],
            value=out.values[0],
        )
