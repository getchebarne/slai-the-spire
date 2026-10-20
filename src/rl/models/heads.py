import math

import torch
import torch.nn as nn


class HeadActionType(nn.Module):
    """L1 ActionType head: x_global gated by the focus context before scoring (AlphaStar-style
    GLU), so the head can mute the global features irrelevant to the current focus."""

    def __init__(self, dim_global: int, dim_ff: int, num_choices: int, dim_context: int) -> None:
        super().__init__()

        self._gate = nn.Linear(dim_context, dim_global)
        self._scorer = nn.Sequential(
            nn.Linear(dim_global, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, num_choices),
        )

    def forward(
        self, x_global: torch.Tensor, x_context: torch.Tensor, x_mask: torch.Tensor
    ) -> torch.Tensor:
        """Score the action-type choices; returns masked logits (sampling/recompute
        is the caller's job)."""
        x_gated = torch.sigmoid(self._gate(x_context)) * x_global
        x_logits = self._scorer(x_gated)
        return x_logits.masked_fill(~x_mask, float("-inf"))


class PointerKeys(nn.Module):
    """
    Entity-class key projection for pointer selection (AlphaStar PointerLogits
    keys net, 1 layer: LayerNorm -> ReLU -> Linear). One instance per entity
    class (cards, potions, ...), shared by every action type that selects that class, so the same
    entity maps to the same key in every selection context.
    """

    def __init__(self, dim_in: int, dim_key: int) -> None:
        super().__init__()

        self._net = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.ReLU(),
            nn.Linear(dim_in, dim_key),
        )

    def forward(self, x_entities: torch.Tensor) -> torch.Tensor:
        """Project entity embeddings (..., dim_in) to pointer keys (..., dim_key)."""
        return self._net(x_entities)


class HeadPointerSelect(nn.Module):
    """Pointer-network selection head (AlphaStar PointerLogits): each entity's logit is
    key·query scaled by 1/√dim_key. The scaling deviates from AlphaStar deliberately:
    unscaled logits move ~√dim_key faster per weight step than the entropy/clip budget expects."""

    def __init__(self, dim_global: int, dim_cond: int, dim_key: int) -> None:
        super().__init__()

        self._scale = 1.0 / math.sqrt(dim_key)
        self._query_net = nn.Sequential(
            nn.LayerNorm(dim_global + dim_cond),
            nn.ReLU(),
            nn.Linear(dim_global + dim_cond, dim_key),
        )

    def forward(
        self,
        x_keys: torch.Tensor,
        x_global: torch.Tensor,
        x_cond: torch.Tensor,
        x_key_state: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score entities by scaled key·query, each key against its state's query; returns
        masked logits. `x_keys` is (..., dim_key) from PointerKeys and `x_key_state` each key's
        state, broadcast to the keys' leading dims; identity dedup is baked into `x_mask`
        (masks.py); sampling is the caller's job."""
        x_query = self._query_net(torch.cat([x_global, x_cond], dim=-1))  # (B, dim_key)
        x_logits = (x_keys * x_query[x_key_state]).sum(-1) * self._scale
        return x_logits.masked_fill(~x_mask, float("-inf"))


class HeadValue(nn.Module):
    """
    Value head for estimating state value (critic in actor-critic).

    One output per reward stream (value decomposition): each output fits its own
    stream's return, and the total state value is the sum over outputs.
    """

    def __init__(self, dim_global: int, dim_ff: int, num_streams: int) -> None:
        super().__init__()

        self._network = nn.Sequential(
            nn.Linear(dim_global, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, num_streams),
        )

    def forward(self, x_global: torch.Tensor) -> torch.Tensor:
        """Estimate per-stream state values."""
        return self._network(x_global)
