import math

import torch
import torch.nn as nn


class HeadActionType(nn.Module):
    """L1 ActionType head: scores each choice from `x_global`.

    Parameters
    ----------
    dim_global : int
        The width of the global vector `x_global`.
    dim_ff : int
        The width of the scorer's hidden layer.
    num_choices : int
        The number of choices the head scores.
    """

    def __init__(self, dim_global: int, dim_ff: int, num_choices: int) -> None:
        super().__init__()

        self._scorer = nn.Sequential(
            nn.Linear(dim_global, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, num_choices),
        )

    def forward(self, x_global: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """Score the action-type choices.

        Sampling/recompute is the caller's job.

        Parameters
        ----------
        x_global : torch.Tensor
            (B, dim_global). Each state's global vector.
        x_mask : torch.Tensor
            (B, NUM_ACTION_TYPES) bool. Each state's legal `ActionType`s.

        Returns
        -------
        torch.Tensor
            (B, num_choices). The masked logits: -inf where `x_mask` is False.
        """
        x_logits = self._scorer(x_global)
        return x_logits.masked_fill(~x_mask, float("-inf"))


class PointerKeys(nn.Module):
    """Token key projection for pointer selection (AlphaStar PointerLogits keys net).

    One layer: LayerNorm -> ReLU -> Linear. One instance, shared by every Entity kind and both
    pointer levels, so the same Entity maps to the same key in every selection context.

    Parameters
    ----------
    dim_in : int
        The width of the Entity embeddings.
    dim_key : int
        The width of the pointer keys.
    """

    def __init__(self, dim_in: int, dim_key: int) -> None:
        super().__init__()

        self._net = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.ReLU(),
            nn.Linear(dim_in, dim_key),
        )

    def forward(self, x_entities: torch.Tensor) -> torch.Tensor:
        """Project Entity embeddings to pointer keys.

        Parameters
        ----------
        x_entities : torch.Tensor
            (..., dim_in). The Entity embeddings.

        Returns
        -------
        torch.Tensor
            (..., dim_key). The pointer keys.
        """
        return self._net(x_entities)


class HeadPointerSelect(nn.Module):
    """Pointer-network selection head (AlphaStar PointerLogits).

    Each Entity's logit is key·query scaled by 1/√dim_key. The scaling deviates from AlphaStar
    deliberately: unscaled logits move ~√dim_key faster per weight step than the entropy/clip
    budget expects.

    Parameters
    ----------
    dim_global : int
        The width of the global vector.
    dim_cond : int
        The width of the conditioning vector the query reads with the global vector.
    dim_key : int
        The width of the keys and the query.
    """

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
        """Score Entities by scaled key·query, each key against its state's query.

        Identity dedup is baked into `x_mask` (src/rl/encoding/mask.py); sampling is the caller's
        job.

        Parameters
        ----------
        x_keys : torch.Tensor
            (..., dim_key). The keys, from PointerKeys.
        x_global : torch.Tensor
            (B, dim_global). Each state's global vector.
        x_cond : torch.Tensor
            (B, dim_cond). Each state's conditioning vector, which its query reads with
            `x_global`.
        x_key_state : torch.Tensor
            (...) long. Each key's state, broadcast to the keys' leading dims.
        x_mask : torch.Tensor
            (...) bool. The legal keys.

        Returns
        -------
        torch.Tensor
            (...). The masked logits: -inf where `x_mask` is False.
        """
        x_query = self._query_net(torch.cat([x_global, x_cond], dim=-1))  # (B, dim_key)
        x_logits = (x_keys * x_query[x_key_state]).sum(-1) * self._scale
        return x_logits.masked_fill(~x_mask, float("-inf"))


class HeadValue(nn.Module):
    """Value head for estimating state value (critic in actor-critic).

    One output per reward stream (value decomposition): each output fits its own stream's return,
    and the total state value is the sum over outputs.

    Parameters
    ----------
    dim_global : int
        The width of the global vector.
    dim_ff : int
        The width of the two hidden layers.
    num_streams : int
        The number of reward streams, one output each.
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
        """Estimate per-stream state values.

        Parameters
        ----------
        x_global : torch.Tensor
            (B, dim_global). Each state's global vector.

        Returns
        -------
        torch.Tensor
            (B, num_streams). Each state's value of each reward stream.
        """
        return self._network(x_global)
