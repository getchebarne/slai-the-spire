"""
Actor-Critic model with hierarchical action heads.

Architecture:
1. Core encoder processes game state → entity embeddings + global context
2. Primary head (action type) selects high-level action type
3. Secondary heads select specific indices (card, monster, map node)
4. Value head estimates state value
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.game.action import Action
from src.game.action import ActionType
from src.game.const import MAP_WIDTH
from src.game.core.fsm import FSM
from src.rl.action_space.cascade import FSM_ROUTING
from src.rl.action_space.cascade import ActionRoute
from src.rl.action_space.cascade import get_secondary_head_type
from src.rl.action_space.types import HeadType
from src.rl.encoding.state import XGameState
from src.rl.models.core import Core
from src.rl.models.core import CoreOutput
from src.rl.models.heads import HeadActionType
from src.rl.models.heads import HeadCardDiscard
from src.rl.models.heads import HeadCardPlay
from src.rl.models.heads import HeadCardRewardSelect
from src.rl.models.heads import HeadCardUpgrade
from src.rl.models.heads import HeadMapSelect
from src.rl.models.heads import HeadMonsterSelect
from src.rl.models.heads import HeadOutput
from src.rl.models.heads import HeadValue


# Maximum number of action types in any FSM state (for action type head output size)
_MAX_ACTION_TYPES = max(len(route.action_types) for route in FSM_ROUTING.values())


@dataclass
class ActorOutput:
    """Output from the actor (policy) - single sample."""

    action_type: ActionType
    action_type_log_prob: torch.Tensor | None  # None if forced (single valid action type)
    secondary_index: int | None  # None if terminal action (no secondary head)
    secondary_log_prob: torch.Tensor | None  # None if terminal action

    def to_action(self) -> Action:
        """Convert to game Action."""
        return Action(type=self.action_type, index=self.secondary_index)

    @property
    def total_log_prob(self) -> torch.Tensor:
        """Combined log probability of the full action."""
        log_probs = []
        if self.action_type_log_prob is not None:
            log_probs.append(self.action_type_log_prob)
        if self.secondary_log_prob is not None:
            log_probs.append(self.secondary_log_prob)

        if not log_probs:
            # Fully deterministic action
            return torch.tensor(0.0)

        return torch.sum(torch.stack(log_probs))


@dataclass
class BatchedActorOutput:
    """Output from the actor (policy) - batched version for parallel environments."""

    # The route determines how to map action_type_indices to ActionType
    route: ActionRoute
    fsm: FSM
    batch_size: int

    # Action type selection (None if forced)
    action_type_indices: torch.Tensor | None  # (B,) indices into route.action_types
    action_type_log_probs: torch.Tensor | None  # (B,) log probs

    # Secondary selection - per sample (None means terminal action for that sample)
    # These tensors have values for ALL samples, but only valid where secondary head was used
    secondary_indices: torch.Tensor | None  # (B,) selected indices
    secondary_log_probs: torch.Tensor | None  # (B,) log probs

    def get_action_type(self, batch_idx: int) -> ActionType:
        """Get action type for a specific sample in the batch."""
        if self.route.is_forced:
            return self.route.forced_action_type
        return self.route.action_types[self.action_type_indices[batch_idx].item()]

    def get_secondary_index(self, batch_idx: int) -> int | None:
        """Get secondary index for a specific sample in the batch."""
        action_type = self.get_action_type(batch_idx)
        secondary_head_type = get_secondary_head_type(self.fsm, action_type)

        # Terminal action - no secondary index
        if secondary_head_type is None:
            return None

        if self.secondary_indices is None:
            return None

        return self.secondary_indices[batch_idx].item()

    def get_action_type_log_prob(self, batch_idx: int) -> torch.Tensor | None:
        """Get action type log prob for a specific sample."""
        if self.action_type_log_probs is None:
            return None
        return self.action_type_log_probs[batch_idx]

    def get_secondary_log_prob(self, batch_idx: int) -> torch.Tensor | None:
        """Get secondary log prob for a specific sample."""
        action_type = self.get_action_type(batch_idx)
        secondary_head_type = get_secondary_head_type(self.fsm, action_type)

        # Terminal action - no secondary log prob
        if secondary_head_type is None:
            return None

        if self.secondary_log_probs is None:
            return None

        return self.secondary_log_probs[batch_idx]

    def to_actor_output(self, batch_idx: int) -> ActorOutput:
        """Extract ActorOutput for a specific sample in the batch."""
        return ActorOutput(
            action_type=self.get_action_type(batch_idx),
            action_type_log_prob=self.get_action_type_log_prob(batch_idx),
            secondary_index=self.get_secondary_index(batch_idx),
            secondary_log_prob=self.get_secondary_log_prob(batch_idx),
        )

    def to_action(self, batch_idx: int) -> Action:
        """Convert to game Action for a specific sample."""
        return Action(
            type=self.get_action_type(batch_idx),
            index=self.get_secondary_index(batch_idx),
        )


@dataclass
class ActorCriticOutput:
    """Full output from actor-critic forward pass - single sample."""

    actor: ActorOutput
    value: torch.Tensor  # (1,) for single sample


@dataclass
class BatchedActorCriticOutput:
    """Full output from actor-critic forward pass - batched version."""

    actor: BatchedActorOutput
    value: torch.Tensor  # (B, 1) for batch


def _slice_core_output(core_out: CoreOutput, indices: torch.Tensor) -> CoreOutput:
    """Slice a CoreOutput to only include specified batch indices."""
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


class ActorCritic(nn.Module):
    """
    Actor-Critic model with hierarchical action heads.

    The action space is hierarchical:
    1. Action type head selects the type of action (play card, end turn, etc.)
    2. Secondary heads select specific indices based on action type
    """

    def __init__(
        self,
        # Core params
        dim_entity: int = 128,
        transformer_dim_ff: int = 256,
        transformer_num_heads: int = 4,
        transformer_num_blocks: int = 2,
        map_encoder_kernel_size: int = 3,
        map_encoder_dim: int = 32,
        # Head params
        dim_ff_action_type: int = 128,
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

        # Primary head: action type selection
        self.head_action_type = HeadActionType(
            dim_global=dim_global,
            dim_ff=dim_ff_action_type,
            max_action_types=_MAX_ACTION_TYPES,
        )

        # Card heads (each with separate weights)
        self.head_card_play = HeadCardPlay(
            dim_entity=dim_entity, dim_global=dim_global, dim_ff=dim_ff_card
        )
        self.head_card_discard = HeadCardDiscard(
            dim_entity=dim_entity, dim_global=dim_global, dim_ff=dim_ff_card
        )
        self.head_card_reward = HeadCardRewardSelect(
            dim_entity=dim_entity, dim_global=dim_global, dim_ff=dim_ff_card
        )
        self.head_card_upgrade = HeadCardUpgrade(
            dim_entity=dim_entity, dim_global=dim_global, dim_ff=dim_ff_card
        )

        # Other secondary heads
        self.head_monster_select = HeadMonsterSelect(
            dim_entity=dim_entity, dim_global=dim_global, dim_ff=dim_ff_monster
        )
        self.head_map_select = HeadMapSelect(
            dim_map=dim_map,
            dim_global=dim_global,
            dim_ff=dim_ff_map,
            num_columns=MAP_WIDTH,
        )

        # Value head (critic)
        self.head_value = HeadValue(dim_global=dim_global, dim_ff=dim_ff_value)

        # Head registry for dispatch
        self._secondary_heads: dict[HeadType, nn.Module] = {
            HeadType.CARD_PLAY: self.head_card_play,
            HeadType.CARD_DISCARD: self.head_card_discard,
            HeadType.CARD_REWARD_SELECT: self.head_card_reward,
            HeadType.CARD_UPGRADE: self.head_card_upgrade,
            HeadType.MONSTER_SELECT: self.head_monster_select,
            HeadType.MAP_SELECT: self.head_map_select,
        }

    def _get_entities_for_head(
        self,
        head_type: HeadType,
        core_out: CoreOutput,
    ) -> torch.Tensor:
        """Get the entity tensor for the given head type."""
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
                raise ValueError(f"No entity tensor for head type: {head_type}")

    def _invoke_secondary_head(
        self,
        head_type: HeadType,
        core_out: CoreOutput,
        mask: torch.Tensor,
        sample: bool,
    ) -> HeadOutput:
        """Invoke the appropriate secondary head."""
        if head_type == HeadType.MAP_SELECT:
            # Map head has different signature
            return self.head_map_select(core_out.x_map, core_out.x_global, mask, sample)

        # Entity selection heads
        head = self._secondary_heads[head_type]
        x_entities = self._get_entities_for_head(head_type, core_out)
        return head(x_entities, core_out.x_global, mask, sample)

    def forward_batch(
        self,
        x_game_state: XGameState,
        fsm: FSM,
        masks: dict[HeadType, torch.Tensor],
        sample: bool = True,
    ) -> BatchedActorCriticOutput:
        """
        Batched forward pass through the actor-critic.

        All samples in the batch must have the same FSM state.
        Different samples may select different action types, and this method
        properly handles running different secondary heads for each sub-group.

        Args:
            x_game_state: Encoded game state (batched)
            fsm: Current FSM state (same for all samples)
            masks: Pre-computed valid action masks for each head type
            sample: Whether to sample actions (True) or just compute logits (False)

        Returns:
            BatchedActorCriticOutput with per-sample actor outputs and values
        """
        # 1. Encode game state
        core_out = self.core(x_game_state)
        batch_size = core_out.x_global.shape[0]
        device = core_out.x_global.device

        # 2. Get routing for this FSM state
        route = FSM_ROUTING[fsm]

        # 3. Action type selection (or bypass if forced)
        if route.is_forced:
            # Single valid action type - skip the action type head
            # All samples get the same action type
            action_type_indices = None
            action_type_log_probs = None
            # Create a tensor of the single action type index for grouping
            per_sample_action_type_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            # Multiple valid action types - run the action type head
            action_type_out = self.head_action_type(
                core_out.x_global, masks[HeadType.ACTION_TYPE], sample
            )
            if sample:
                action_type_indices = action_type_out.indices  # (B,)
                action_type_log_probs = action_type_out.log_probs  # (B,)
                per_sample_action_type_idx = action_type_indices
            else:
                action_type_indices = None
                action_type_log_probs = None
                per_sample_action_type_idx = None

        # 4. Secondary heads - handle different action types per sample
        # Initialize output tensors (will be filled in per-group)
        secondary_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        secondary_log_probs = torch.zeros(batch_size, device=device)
        has_any_secondary = False

        if sample and per_sample_action_type_idx is not None:
            # Group samples by their selected action type
            unique_action_type_idxs = torch.unique(per_sample_action_type_idx)

            for action_type_idx in unique_action_type_idxs:
                action_type_idx_item = action_type_idx.item()
                action_type = route.action_types[action_type_idx_item]

                # Find which samples chose this action type
                sample_mask = per_sample_action_type_idx == action_type_idx
                sample_indices = torch.where(sample_mask)[0]

                # Get secondary head type for this action type
                secondary_head_type = get_secondary_head_type(fsm, action_type)

                if secondary_head_type is None:
                    # Terminal action - no secondary head needed for these samples
                    continue

                has_any_secondary = True

                # Slice core output and masks for this subset
                subset_core_out = _slice_core_output(core_out, sample_indices)
                subset_mask = masks[secondary_head_type][sample_indices]

                # Run secondary head on subset
                subset_out = self._invoke_secondary_head(
                    secondary_head_type, subset_core_out, subset_mask, sample=True
                )

                # Scatter results back to original positions
                secondary_indices[sample_indices] = subset_out.indices
                secondary_log_probs[sample_indices] = subset_out.log_probs

        # Convert to None if no secondary heads were used
        if not has_any_secondary:
            secondary_indices = None
            secondary_log_probs = None

        # 5. Value estimate
        value = self.head_value(core_out.x_global)

        actor_output = BatchedActorOutput(
            route=route,
            fsm=fsm,
            batch_size=batch_size,
            action_type_indices=action_type_indices,
            action_type_log_probs=action_type_log_probs,
            secondary_indices=secondary_indices,
            secondary_log_probs=secondary_log_probs,
        )

        return BatchedActorCriticOutput(actor=actor_output, value=value)

    def forward(
        self,
        x_game_state: XGameState,
        fsm: FSM,
        masks: dict[HeadType, torch.Tensor],
        sample: bool = True,
    ) -> ActorCriticOutput:
        """
        Forward pass for a single sample.

        Args:
            x_game_state: Encoded game state (batch size 1)
            fsm: Current FSM state
            masks: Pre-computed valid action masks for each head type
            sample: Whether to sample actions (True) or just compute logits (False)

        Returns:
            ActorCriticOutput with actor output and value estimate
        """
        batched_output = self.forward_batch(x_game_state, fsm, masks, sample)

        # Extract single sample result
        actor_output = batched_output.actor.to_actor_output(0)
        value = batched_output.value[0] if batched_output.value.dim() > 1 else batched_output.value

        return ActorCriticOutput(actor=actor_output, value=value)

    def get_action(
        self,
        x_game_state: XGameState,
        fsm: FSM,
        masks: dict[HeadType, torch.Tensor],
    ) -> tuple[Action, ActorOutput, torch.Tensor]:
        """
        Convenience method to get an action for a single game state.

        Returns:
            Tuple of (Action, ActorOutput, value)
        """
        output = self.forward(x_game_state, fsm, masks, sample=True)
        action = output.actor.to_action()
        return action, output.actor, output.value
