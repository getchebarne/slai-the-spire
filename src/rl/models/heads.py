"""
Action heads for the Actor-Critic model.

This module contains all the heads used to select actions:
- HeadActionType: Selects high-level action type (play card, end turn, etc.)
- HeadEntitySelection: Base class for selecting one entity from a sequence
- HeadCardPlay, HeadCardDiscard, HeadCardReward, HeadCardUpgrade: Card selection
- HeadMonsterSelect: Monster targeting
- HeadMapSelect: Map node selection
- HeadValue: State value estimation (critic)
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class HeadOutput:
    logits: torch.Tensor  # (B, num_options) raw scores before masking
    indices: torch.Tensor | None  # (B,) sampled indices, None if not sampling
    log_probs: torch.Tensor | None  # (B,) log probs of sampled indices, None if not sampling


def sample_from_logits(
    logits: torch.Tensor,
    mask: torch.Tensor,
    sample: bool,
) -> HeadOutput:
    """
    Apply mask to logits and optionally sample.

    Args:
        logits: Raw scores (B, num_options)
        mask: Valid action mask (B, num_options), True = valid
        sample: Whether to sample an action

    Returns:
        HeadOutput with logits, and optionally indices/log_probs
    """
    # Mask invalid actions
    masked_logits = logits.masked_fill(~mask, float("-inf"))

    if not sample:
        return HeadOutput(logits=masked_logits, indices=None, log_probs=None)

    # Sample from categorical distribution
    dist = torch.distributions.Categorical(logits=masked_logits)
    indices = dist.sample()
    log_probs = dist.log_prob(indices)

    return HeadOutput(logits=masked_logits, indices=indices, log_probs=log_probs)


# =============================================================================
# Action Type Head (Primary)
# =============================================================================


class HeadActionType(nn.Module):
    """
    Primary head that selects the action type.

    This head chooses between high-level action types (e.g., play card vs end turn).
    The number of outputs is dynamic based on the FSM state.
    """

    def __init__(self, dim_global: int, dim_ff: int, max_action_types: int):
        """
        Args:
            dim_global: Dimension of the global context vector
            dim_ff: Hidden dimension of the feedforward network
            max_action_types: Maximum number of action types (for output layer size)
        """
        super().__init__()

        self._scorer = nn.Sequential(
            nn.Linear(dim_global, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, max_action_types),
        )

    def forward(
        self,
        x_global: torch.Tensor,
        mask: torch.Tensor,
        sample: bool = True,
    ) -> HeadOutput:
        """
        Score and optionally sample action type.

        Args:
            x_global: Global context vector (B, dim_global)
            mask: Valid action type mask (B, num_action_types), True = valid
            sample: Whether to sample an action

        Returns:
            HeadOutput with scores and optionally sampled indices
        """
        logits = self._scorer(x_global)  # (B, max_action_types)

        # Slice to match mask size (for FSM states with fewer action types)
        logits = logits[:, : mask.shape[1]]

        return sample_from_logits(logits, mask, sample)


# =============================================================================
# Entity Selection Heads (Secondary)
# =============================================================================


class HeadEntitySelection(nn.Module):
    """
    Base class for heads that select one entity from a sequence.
    Used for cards, monsters, etc.
    """

    def __init__(self, dim_entity: int, dim_global: int, dim_ff: int):
        """
        Args:
            dim_entity: Dimension of each entity embedding
            dim_global: Dimension of the global context vector
            dim_ff: Hidden dimension of the feedforward network
        """
        super().__init__()

        self._scorer = nn.Sequential(
            nn.Linear(dim_entity + dim_global, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, 1),
        )

    def forward(
        self,
        x_entities: torch.Tensor,
        x_global: torch.Tensor,
        mask: torch.Tensor,
        sample: bool = True,
    ) -> HeadOutput:
        """
        Score and optionally sample from entities.

        Args:
            x_entities: Entity embeddings (B, N, dim_entity)
            x_global: Global context vector (B, dim_global)
            mask: Valid entity mask (B, N), True = valid
            sample: Whether to sample an action

        Returns:
            HeadOutput with scores and optionally sampled indices
        """
        _, num_entities, _ = x_entities.shape

        # Broadcast global context to each entity
        x_global_exp = torch.unsqueeze(x_global, 1).expand(-1, num_entities, -1)
        x_input = torch.cat([x_entities, x_global_exp], dim=-1)

        # Score each entity
        logits = torch.squeeze(self._scorer(x_input), -1)  # (B, N)

        return sample_from_logits(logits, mask, sample)


class HeadCardPlay(HeadEntitySelection):
    """Head for selecting a card from hand to play."""

    pass


class HeadCardDiscard(HeadEntitySelection):
    """Head for selecting a card from hand to discard."""

    pass


class HeadCardRewardSelect(HeadEntitySelection):
    """Head for selecting a card from combat rewards."""

    pass


class HeadCardUpgrade(HeadEntitySelection):
    """Head for selecting a card from deck to upgrade at rest sites."""

    pass


class HeadMonsterSelect(HeadEntitySelection):
    """Head for selecting a monster to target."""

    pass


# =============================================================================
# Map Selection Head
# =============================================================================


class HeadMapSelect(nn.Module):
    """
    Head for selecting the next map node.

    Unlike entity selection heads, this operates on the encoded map representation
    rather than a sequence of entity embeddings.
    """

    def __init__(self, dim_map: int, dim_global: int, dim_ff: int, num_columns: int):
        """
        Args:
            dim_map: Dimension of the map encoding
            dim_global: Dimension of the global context vector
            dim_ff: Hidden dimension of the feedforward network
            num_columns: Number of map columns (MAP_WIDTH)
        """
        super().__init__()

        self._scorer = nn.Sequential(
            nn.Linear(dim_map + dim_global, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, num_columns),
        )

    def forward(
        self,
        x_map: torch.Tensor,
        x_global: torch.Tensor,
        mask: torch.Tensor,
        sample: bool = True,
    ) -> HeadOutput:
        """
        Score and optionally sample map node.

        Args:
            x_map: Map encoding (B, dim_map)
            x_global: Global context vector (B, dim_global)
            mask: Valid node mask (B, num_columns), True = valid
            sample: Whether to sample an action

        Returns:
            HeadOutput with scores and optionally sampled indices
        """
        x_input = torch.cat([x_map, x_global], dim=-1)
        logits = self._scorer(x_input)  # (B, num_columns)

        return sample_from_logits(logits, mask, sample)


# =============================================================================
# Value Head (Critic)
# =============================================================================


class HeadValue(nn.Module):
    """
    Value head for estimating state value (critic in actor-critic).

    Outputs a single scalar value estimate for the current state.
    """

    def __init__(self, dim_global: int, dim_ff: int):
        """
        Args:
            dim_global: Dimension of the global context vector
            dim_ff: Hidden dimension of the feedforward network
        """
        super().__init__()

        self._network = nn.Sequential(
            nn.Linear(dim_global, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, 1),
        )

    def forward(self, x_global: torch.Tensor) -> torch.Tensor:
        """
        Estimate state value.

        Args:
            x_global: Global context vector (B, dim_global)

        Returns:
            Value estimate (B, 1)
        """
        return self._network(x_global)
