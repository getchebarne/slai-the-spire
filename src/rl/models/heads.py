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
# Entity Selection Heads (Secondary)
# =============================================================================


class HeadEntitySelection(nn.Module):
    """
    Selects one entity from a sequence (cards, relics, ...). One instance per Pool,
    shared across the operations on that pool and conditioned on an operation
    embedding (AlphaStar-style: shared selection network, per-operation conditioning).
    """

    def __init__(self, dim_entity: int, dim_global: int, dim_op: int, dim_ff: int):
        """
        Args:
            dim_entity: Dimension of each entity embedding
            dim_global: Dimension of the global context vector
            dim_op: Dimension of the operation-conditioning embedding
            dim_ff: Hidden dimension of the feedforward network
        """
        super().__init__()

        self._scorer = nn.Sequential(
            nn.Linear(dim_entity + dim_global + dim_op, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, 1),
        )

    def forward(
        self,
        x_entities: torch.Tensor,
        x_global: torch.Tensor,
        x_op: torch.Tensor,
        mask: torch.Tensor,
    ) -> HeadOutput:
        """Score each entity conditioned on the operation; returns masked logits.
        Identity dedup is baked into `mask` (masks.py); sampling is the caller's job."""
        _, num_entities, _ = x_entities.shape
        x_global_exp = torch.unsqueeze(x_global, 1).expand(-1, num_entities, -1)
        x_op_exp = torch.unsqueeze(x_op, 1).expand(-1, num_entities, -1)
        x_input = torch.cat([x_entities, x_global_exp, x_op_exp], dim=-1)
        logits = torch.squeeze(self._scorer(x_input), -1)  # (B, N)
        return HeadOutput(logits.masked_fill(~mask, float("-inf")), None, None)


class HeadMonsterSelect(nn.Module):
    """
    Head for selecting a monster to target.

    Unlike other entity selection heads, this receives the active entity embedding
    (the played card or thrown potion) as an additional input so the model can make
    source-dependent targeting decisions.
    Input per monster: [monster_emb, global, active_emb].
    """

    def __init__(self, dim_entity: int, dim_global: int, dim_ff: int):
        super().__init__()

        self._scorer = nn.Sequential(
            nn.Linear(dim_entity + dim_global + dim_entity, dim_ff),
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
        x_active_card: torch.Tensor,
    ) -> HeadOutput:
        """Score each monster conditioned on the active entity; returns masked logits.
        `x_active_card` is the played card for CardPlay, the thrown potion for PotionUse."""
        _, num_entities, _ = x_entities.shape
        x_global_exp = torch.unsqueeze(x_global, 1).expand(-1, num_entities, -1)
        x_active_exp = torch.unsqueeze(x_active_card, 1).expand(-1, num_entities, -1)
        x_input = torch.cat([x_entities, x_global_exp, x_active_exp], dim=-1)
        logits = torch.squeeze(self._scorer(x_input), -1)  # (B, N)
        return HeadOutput(logits.masked_fill(~mask, float("-inf")), None, None)


# =============================================================================
# Map Selection Head
# =============================================================================


class HeadMapSelect(nn.Module):
    """
    Head for selecting the next map node: one logit per map column, scored from
    that column's embedding (weight-shared across columns, like the entity heads).
    """

    def __init__(self, dim_map: int, dim_global: int, dim_ff: int):
        """
        Args:
            dim_map: Dimension of each per-column map embedding
            dim_global: Dimension of the global context vector
            dim_ff: Hidden dimension of the feedforward network
        """
        super().__init__()

        self._scorer = nn.Sequential(
            nn.Linear(dim_map + dim_global, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, 1),
        )

    def forward(
        self,
        x_map: torch.Tensor,
        x_global: torch.Tensor,
        mask: torch.Tensor,
    ) -> HeadOutput:
        """Score each map column; returns masked logits (no dedup — columns are
        distinct paths). `x_map` is (B, MAP_WIDTH, dim_map); sampling/recompute
        is the caller's job."""
        _, num_columns, _ = x_map.shape
        x_global_exp = torch.unsqueeze(x_global, 1).expand(-1, num_columns, -1)
        x_input = torch.cat([x_map, x_global_exp], dim=-1)
        logits = torch.squeeze(self._scorer(x_input), -1)  # (B, num_columns)
        return HeadOutput(logits.masked_fill(~mask, float("-inf")), None, None)


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
