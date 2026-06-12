import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class HeadOutput:
    logits: torch.Tensor  # (B, num_options) raw scores after masking
    indices: torch.Tensor | None  # (B,) sampled indices, None if not sampling
    log_probs: torch.Tensor | None  # (B,) log probs of sampled indices, None if not sampling


# =============================================================================
# Binary Choice Head (for decision primaries)
# =============================================================================


class HeadBinaryChoice(nn.Module):
    """
    L1 action-kind head: x_global gated by the screen context before scoring
    (AlphaStar-style GLU — the action space is modal per screen, so the gate lets
    the head mute the global features irrelevant to the current mode).
    """

    def __init__(self, dim_global: int, dim_ff: int, num_choices: int, dim_context: int):
        super().__init__()

        self._gate = nn.Linear(dim_context, dim_global)
        self._scorer = nn.Sequential(
            nn.Linear(dim_global, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, num_choices),
        )

    def forward(
        self, x_global: torch.Tensor, context: torch.Tensor, mask: torch.Tensor
    ) -> HeadOutput:
        """Score the option-kind choices; returns masked logits (sampling/recompute
        is the caller's job)."""
        gated = torch.sigmoid(self._gate(context)) * x_global
        logits = self._scorer(gated)
        return HeadOutput(logits.masked_fill(~mask, float("-inf")), None, None)


# =============================================================================
# Pointer Selection Heads (Secondary + Target)
# =============================================================================


class PointerKeys(nn.Module):
    """
    Entity-class key projection for pointer selection (AlphaStar PointerLogits
    keys net, 1 layer: LayerNorm -> ReLU -> Linear). One instance per entity
    class (cards, potions, ...), shared by every pool of that class, so the same
    entity maps to the same key in every selection context.
    """

    def __init__(self, dim_in: int, dim_key: int):
        super().__init__()

        self._net = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.ReLU(),
            nn.Linear(dim_in, dim_key),
        )

    def forward(self, x_entities: torch.Tensor) -> torch.Tensor:
        """Project entity embeddings (B, N, dim_in) to pointer keys (B, N, dim_key)."""
        return self._net(x_entities)


class HeadPointerSelect(nn.Module):
    """
    Pointer-network selection head (AlphaStar PointerLogits): the decision context
    projects to a query (1 layer: LayerNorm -> ReLU -> Linear), and each entity's
    logit is the key·query dot product scaled by 1/√dim_key. The scaling deviates
    from AlphaStar (raw dot products) deliberately: the dot product sums dim_key
    unit-scale terms, so unscaled logits move ~√dim_key faster per weight step
    than the concat-MLP heads the entropy/clip budget was tuned on — measured as
    2× approx_kl and 2× faster entropy collapse on NEWERA-VIII. Selection is
    retrieval in the shared key space: keys encode entity qualities once, the
    query encodes what the current decision wants.
    """

    def __init__(self, dim_global: int, dim_cond: int, dim_key: int):
        """
        Args:
            dim_global: Dimension of the global context vector
            dim_cond: Dimension of the conditioning vector (operation embedding
                for L2, active-entity embedding for L3)
            dim_key: Dimension of the pointer keys/query
        """
        super().__init__()

        self._scale = 1.0 / math.sqrt(dim_key)
        self._query_net = nn.Sequential(
            nn.LayerNorm(dim_global + dim_cond),
            nn.ReLU(),
            nn.Linear(dim_global + dim_cond, dim_key),
        )

    def forward(
        self,
        keys: torch.Tensor,
        x_global: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor,
    ) -> HeadOutput:
        """Score entities by scaled key·query; returns masked logits. `keys` is
        (B, N, dim_key) from PointerKeys; identity dedup is baked into `mask`
        (masks.py); sampling is the caller's job."""
        query = self._query_net(torch.cat([x_global, cond], dim=-1))  # (B, dim_key)
        logits = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1) * self._scale  # (B, N)
        return HeadOutput(logits.masked_fill(~mask, float("-inf")), None, None)


# =============================================================================
# Value Head (Critic)
# =============================================================================


class HeadValue(nn.Module):
    """
    Value head for estimating state value (critic in actor-critic).

    One output per reward stream (value decomposition): each output fits its own
    stream's return, and the total state value is the sum over outputs.
    """

    def __init__(self, dim_global: int, dim_ff: int, num_streams: int):
        """
        Args:
            dim_global: Dimension of the global context vector
            dim_ff: Hidden dimension of the feedforward network
            num_streams: Number of reward streams (len(REWARD_STREAMS))
        """
        super().__init__()

        self._network = nn.Sequential(
            nn.Linear(dim_global, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, num_streams),
        )

    def forward(self, x_global: torch.Tensor) -> torch.Tensor:
        """
        Estimate per-stream state values.

        Args:
            x_global: Global context vector (B, dim_global)

        Returns:
            Value estimates (B, num_streams)
        """
        return self._network(x_global)
