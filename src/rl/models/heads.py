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
    logits: torch.Tensor  # (B, num_options) raw scores after masking
    indices: torch.Tensor | None  # (B,) sampled indices, None if not sampling
    log_probs: torch.Tensor | None  # (B,) log probs of sampled indices, None if not sampling


# =============================================================================
# Grouped Sampling
# =============================================================================
#
# Identical cards produce identical logits (same encoding → same embedding
# → same score). Rather than splitting probability mass across duplicates,
# we deduplicate: softmax over one representative per unique logit value,
# then randomly pick an instance within the chosen group.
#
# Example:
#   Hand: [Strike, Strike, Defend]   logits: [0.5, 0.5, 0.8]
#   Representatives: Strike(0.5), Defend(0.8)
#   Grouped softmax: Strike=43%, Defend=57%
#   If Strike is sampled → randomly pick one of the two Strikes.
#
# This gives cleaner gradients (the model learns "play Strike", not
# "play card in slot 0") and consistent sampling/greedy behavior.
# =============================================================================


def _get_representative_mask(masked_logits: torch.Tensor) -> torch.Tensor:
    """
    For groups of identical valid logits, mark only the first occurrence.

    Args:
        masked_logits: (B, N) with -inf for invalid positions

    Returns:
        Boolean mask (B, N), True = first valid occurrence of its logit value
    """
    N = masked_logits.shape[1]
    device = masked_logits.device

    valid = masked_logits != float("-inf")  # (B, N)

    # same_logit[b, i, j] = True if positions i and j share a logit value
    same_logit = torch.unsqueeze(masked_logits, 2) == torch.unsqueeze(masked_logits, 1)

    # earlier[i, j] = True if j < i (strict lower-triangular)
    earlier = torch.tril(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=-1)

    # has_earlier_dup[b, i] = exists valid j < i with same logit as i
    has_earlier_dup = torch.any(
        same_logit & earlier.unsqueeze(0) & valid.unsqueeze(1),
        dim=2,
    )

    return valid & ~has_earlier_dup


def _sample_grouped(masked_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample from groups of identical logits.

    1. Deduplicate: keep one representative per unique logit value
    2. Softmax over representatives → group probabilities
    3. Sample a group
    4. Uniformly pick an instance within the chosen group

    log_prob = log(group_probability). The within-group uniform selection
    is not part of the policy (identical cards are interchangeable).

    Args:
        masked_logits: (B, N) with -inf for invalid positions

    Returns:
        (indices, log_probs): selected position indices (B,) and group log probs (B,)
    """
    B = masked_logits.shape[0]
    device = masked_logits.device

    # Deduplicate: softmax over one representative per unique logit
    is_rep = _get_representative_mask(masked_logits)
    rep_logits = masked_logits.masked_fill(~is_rep, float("-inf"))

    # Sample a group (via its representative)
    group_dist = torch.distributions.Categorical(logits=rep_logits)
    rep_idx = group_dist.sample()
    log_probs = group_dist.log_prob(rep_idx)

    # Find all valid members of the sampled group
    rep_values = masked_logits[torch.arange(B, device=device), rep_idx]
    valid = masked_logits != float("-inf")
    in_group = (masked_logits == torch.unsqueeze(rep_values, 1)) & valid

    # Uniformly pick one instance from the group
    uniform_probs = in_group.float()
    uniform_probs = uniform_probs / torch.sum(uniform_probs, dim=1, keepdim=True)
    indices = torch.distributions.Categorical(probs=uniform_probs).sample()

    return indices, log_probs


def compute_grouped_log_prob_and_entropy(
    masked_logits: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute grouped log probability and entropy for given action indices.

    Used during PPO recomputation to evaluate actions under the current policy.
    Maps each index to its group's representative, then evaluates the grouped
    distribution.

    Args:
        masked_logits: (B, N) with -inf for invalid positions
        indices: (B,) position indices of actions taken

    Returns:
        (log_probs, entropy) of the grouped distribution, both (B,)
    """
    B = masked_logits.shape[0]
    device = masked_logits.device

    is_rep = _get_representative_mask(masked_logits)
    rep_logits = masked_logits.masked_fill(~is_rep, float("-inf"))
    group_dist = torch.distributions.Categorical(logits=rep_logits)

    # Map each index to its group's representative
    idx_logits = masked_logits[torch.arange(B, device=device), indices]
    rep_for_idx = is_rep & (masked_logits == torch.unsqueeze(idx_logits, 1))
    rep_indices = torch.argmax(rep_for_idx.int(), dim=1)

    return group_dist.log_prob(rep_indices), group_dist.entropy()


def get_grouped_probs(masked_logits: torch.Tensor) -> torch.Tensor:
    """
    Compute per-position probabilities from the grouped distribution.

    Each position gets its group's probability. Identical cards (same logit)
    share the same probability value.

    Args:
        masked_logits: (B, N) with -inf for invalid positions

    Returns:
        (B, N) probabilities
    """
    is_rep = _get_representative_mask(masked_logits)
    rep_logits = masked_logits.masked_fill(~is_rep, float("-inf"))
    rep_probs = torch.softmax(rep_logits, dim=-1)

    # Propagate each representative's probability to all group members
    same_logit = torch.unsqueeze(masked_logits, 2) == torch.unsqueeze(masked_logits, 1)
    return torch.sum(same_logit.float() * torch.unsqueeze(rep_probs, 1), dim=2)


def sample_from_logits(
    logits: torch.Tensor,
    mask: torch.Tensor,
    sample: bool,
) -> HeadOutput:
    """
    Apply mask to logits and optionally sample using grouped distribution.

    Grouped sampling deduplicates identical logits (from identical cards)
    so the model samples over card *types*, not card *slots*.

    Args:
        logits: Raw scores (B, num_options)
        mask: Valid action mask (B, num_options), True = valid
        sample: Whether to sample an action

    Returns:
        HeadOutput with masked logits, and optionally indices/log_probs
    """
    # Mask invalid actions
    masked_logits = logits.masked_fill(~mask, float("-inf"))

    if not sample:
        return HeadOutput(logits=masked_logits, indices=None, log_probs=None)

    # Grouped sampling: deduplicate identical logits, sample group, pick instance
    indices, log_probs = _sample_grouped(masked_logits)

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
