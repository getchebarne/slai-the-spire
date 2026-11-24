import math
from typing import cast

import torch
import torch.nn as nn
from slai import ACTION_SPEC_REGISTRY

from src.rl.batches import get_entity_kind_rows
from src.rl.constants import ACTION_TYPES
from src.rl.constants import NUM_ACTION_TYPES
from src.rl.types import CoreOutput
from src.rl.types import EntityKind


# The step of the value head's running mean and variance toward each rollout's, per stream
_RETURN_MOMENT_STEP = 0.03

# The floor of the variance the streams share, so that the value head's scale σ is at least 1
_MIN_RETURN_VARIANCE = 1.0


class HeadActionType(nn.Module):
    """L1 ActionType head: scores each choice from `x_global`.

    Parameters
    ----------
    dim_global : int
        The width of the global vector `x_global`.
    dim_ff : int
        The width of the scorer's hidden layer.
    """

    def __init__(self, dim_global: int, dim_ff: int) -> None:
        super().__init__()

        self._scorer = nn.Sequential(
            nn.Linear(dim_global, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, NUM_ACTION_TYPES),
        )

    def init_logit_weights(self) -> None:
        """Initialize the logit layer near zero after the general linear init."""
        nn.init.orthogonal_(self._scorer[-1].weight, gain=0.01)

    def forward(self, core_output: CoreOutput) -> torch.Tensor:
        """Score the action-type choices.

        Sampling/recompute is the caller's job.

        Parameters
        ----------
        core_output : CoreOutput
            The core's output.

        Returns
        -------
        torch.Tensor
            (B, NUM_ACTION_TYPES). The masked logits: -inf where the `ActionType` is illegal.
        """
        x_logit = self._scorer(core_output.x_global)

        return x_logit.masked_fill(~core_output.x_legal_action_type, float("-inf"))


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

    def forward(self, x_entity: torch.Tensor) -> torch.Tensor:
        """Project Entity embeddings to pointer keys.

        Parameters
        ----------
        x_entity : torch.Tensor
            (..., dim_in). The Entity embeddings.

        Returns
        -------
        torch.Tensor
            (..., dim_key). The pointer keys.
        """
        return self._net(x_entity)


class PointerLogits(nn.Module):
    """Pointer-network logits (AlphaStar PointerLogits), the scoring both pointer heads share.

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

    def init_logit_weights(self) -> None:
        """Initialize the logit layer near zero after the general linear init."""
        nn.init.orthogonal_(self._query_net[-1].weight, gain=0.01)

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
        x_logit = (x_keys * x_query[x_key_state]).sum(-1) * self._scale

        return x_logit.masked_fill(~x_mask, float("-inf"))


class HeadSelection(nn.Module):
    """L2 selection head: scores each token of a state as its `ActionType`'s selection.

    Parameters
    ----------
    dim_global : int
        The width of the global vector.
    dim_op : int
        The width of the `ActionType` embedding.
    dim_key : int
        The width of the pointer keys and the query.
    """

    _x_has_selection: torch.Tensor

    def __init__(self, dim_global: int, dim_op: int, dim_key: int) -> None:
        super().__init__()

        # `ActionType` embedding for the L2 pointer selection
        self._action_type_emb = nn.Embedding(NUM_ACTION_TYPES, dim_op)
        self._pointer_logits = PointerLogits(dim_global, dim_op, dim_key)

        # Buffer for each `ActionType`'s L2 requirement
        self.register_buffer(
            "_x_has_selection",
            torch.tensor(
                [ACTION_SPEC_REGISTRY[action_type].arity != (0, 0) for action_type in ACTION_TYPES]
            ),
            persistent=False,
        )

    def init_logit_weights(self) -> None:
        """Initialize the logit layer near zero after the general linear init."""
        self._pointer_logits.init_logit_weights()

    def forward(
        self, core_output: CoreOutput, x_token_key: torch.Tensor, x_action_type: torch.Tensor
    ) -> torch.Tensor:
        """Score each token as the selection of its state's action type.

        Parameters
        ----------
        core_output : CoreOutput
            The core's output.
        x_token_key : torch.Tensor
            (N, dim_key). Each token's key, from PointerKeys.
        x_action_type : torch.Tensor
            (B,) long. Each state's action type, the L1 selection.

        Returns
        -------
        torch.Tensor
            (N,). Each token's masked logit: -inf where its state's action type can't select it.
            Where the action type selects no token, its state's Character is the one legal row.
        """
        tokens_refined = core_output.tokens
        x_token_state = core_output.x_token_state

        # Wether each token is selectable under its state selected `ActionType`
        x_token_selectable = (
            (tokens_refined.x_legal_selection >> x_action_type[x_token_state]) & 1
        ).bool()  # (N,)

        # An `ActionType` that selects no token sets its state's Character as legal to avoid NaNs
        x_is_character = tokens_refined.x_entity_kind == EntityKind.CHARACTER  # (N,)
        x_has_selection = self._x_has_selection[x_action_type]
        x_token_selectable |= x_is_character & ~x_has_selection[x_token_state]

        # Calculate the rows' logits
        return self._pointer_logits(
            x_token_key,
            core_output.x_global,
            self._action_type_emb(x_action_type),
            x_token_state,
            x_token_selectable,
        )


class HeadTarget(nn.Module):
    """L3 target head: scores each Monster of a state as its selected token's target.

    Parameters
    ----------
    dim_global : int
        The width of the global vector.
    dim_entity : int
        The width of the core's refined token rows.
    dim_key : int
        The width of the pointer keys and the query.
    """

    _x_has_target: torch.Tensor

    def __init__(self, dim_global: int, dim_entity: int, dim_key: int) -> None:
        super().__init__()

        self._dim_key = dim_key
        self._pointer_logits = PointerLogits(dim_global, dim_entity, dim_key)

        # Buffer for each `ActionType`'s L3 requirement
        self.register_buffer(
            "_x_has_target",
            torch.tensor(
                [ACTION_SPEC_REGISTRY[action_type].arity[1] == 2 for action_type in ACTION_TYPES]
            ),
            persistent=False,
        )

    def init_logit_weights(self) -> None:
        """Initialize the logit layer near zero after the general linear init."""
        self._pointer_logits.init_logit_weights()

    def forward(
        self,
        core_output: CoreOutput,
        x_token_key: torch.Tensor,
        x_target_state: torch.Tensor,
        x_action_type: torch.Tensor,
        x_selection_row: torch.Tensor,
    ) -> torch.Tensor:
        """Score each Monster as the target of its state's selected token, then "no target".

        Parameters
        ----------
        core_output : CoreOutput
            The core's output.
        x_token_key : torch.Tensor
            (N, dim_key). Each token's key, from PointerKeys.
        x_target_state : torch.Tensor
            (M + B,) long. Each target row's state: the Monsters', then each state's "no target".
        x_action_type : torch.Tensor
            (B,) long. Each state's action type, the L1 selection.
        x_selection_row : torch.Tensor
            (B,) long. Each state's selected token's row among the tokens, its Character's where
            none: the L2 selection.

        Returns
        -------
        torch.Tensor
            (M + B,). Each target row's masked logit, the Monsters' and then each state's "no
            target": -inf where illegal. Where the selection takes no target, its state's "no
            target" is the one legal row.
        """
        tokens = core_output.tokens
        batch_size = x_action_type.shape[0]

        # The target rows are the M Monsters, then one "no target" per state
        num_monsters = x_target_state.shape[0] - batch_size

        # The Monsters' rows in the window, state after state
        x_monster_row = get_entity_kind_rows(
            tokens.x_entity_kind, EntityKind.MONSTER, num_monsters
        )  # (M,)

        # Each Monster's state: the target rows' first M
        x_monster_state = x_target_state[:num_monsters]  # (M,)

        # For actions that take one (CardPlay / PotionUse) and tokens that need one only
        x_has_target = self._x_has_target[x_action_type] & tokens.x_needs_target[x_selection_row] # (B,)
        x_target_legal = (
            (tokens.x_legal_target[x_monster_row] >> x_action_type[x_monster_state]) & 1
        ).bool()  # (M,)
        x_mask_l3 = torch.cat([x_has_target[x_monster_state] & x_target_legal, ~x_has_target]) # (M + B,)

        # A cat, not F.pad: compiled, F.pad's backward fails on an empty Monster block
        x_keys = torch.cat(
            [
                x_token_key[x_monster_row],
                torch.zeros(
                    batch_size,
                    self._dim_key,
                    dtype=tokens.features.x_value.dtype,
                    device=tokens.features.x_value.device,
                ),
            ]
        )

        # Calculate the target rows' logits
        return self._pointer_logits(
            x_keys,
            core_output.x_global,
            tokens.features.x_value[x_selection_row],
            x_target_state,
            x_mask_l3,
        )


class HeadValue(nn.Module):
    """Value head for estimating state value (critic in actor-critic).

    One output per reward stream (value decomposition): each output fits its own stream's return,
    and the total state value is the sum over outputs.

    The network fits normalized returns (PopArt): a value is σ times the network's output plus its
    stream's running mean of the return targets, with one σ ≥ 1 shared by the streams.
    `update_value_statistics` steps those statistics, preserving every output.

    Parameters
    ----------
    dim_global : int
        The width of the global vector.
    dim_ff : int
        The width of the two hidden layers.
    num_streams : int
        The number of reward streams, one output each.
    """

    _x_return_mean: torch.Tensor
    _x_return_variance: torch.Tensor

    def __init__(self, dim_global: int, dim_ff: int, num_streams: int) -> None:
        super().__init__()

        self._network = nn.Sequential(
            nn.Linear(dim_global, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, num_streams),
        )

        # Buffers of each stream's running mean and variance of the return targets
        self.register_buffer("_x_return_mean", torch.zeros(num_streams), persistent=True)
        self.register_buffer("_x_return_variance", torch.ones(num_streams), persistent=True)

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
        return _get_value_scale(self) * self._network(x_global) + self._x_return_mean


def update_value_statistics(
    head_value: HeadValue,
    x_return_mean_rollout: torch.Tensor,
    x_return_second_moment_rollout: torch.Tensor,
) -> torch.Tensor:
    """Step the value head's statistics toward a rollout's moments of the returns.

    Each stream's running mean takes one step toward the rollout's mean, and its running variance
    one step toward the rollout's variance, the second moment minus the mean squared. The last
    layer is then rescaled so that every output, σ times the network's plus μ, is unchanged under
    the new σ and μ (PopArt). Adam's state is left as it is.

    Parameters
    ----------
    head_value : HeadValue
        The value head, its statistics and its last layer updated in place.
    x_return_mean_rollout : torch.Tensor
        (num_streams,). Each stream's mean of the rollout's returns over every rank's states, the
        same on every rank.
    x_return_second_moment_rollout : torch.Tensor
        (num_streams,). Each stream's mean of the rollout's squared returns over every rank's
        states, the same on every rank.

    Returns
    -------
    torch.Tensor
        (). The value head's scale σ after the step, on its device.
    """
    layer = cast(nn.Linear, head_value._network[-1])
    with torch.no_grad():
        # σ and μ before the step
        x_value_scale_old = _get_value_scale(head_value)
        x_return_mean_old = head_value._x_return_mean.clone()

        # Each stream's variance over the rollout's states
        x_return_variance_rollout = x_return_second_moment_rollout - x_return_mean_rollout.square()

        # Step each mean and variance toward the rollout's: x ← (1 − step)·x + step·x_rollout
        head_value._x_return_mean.lerp_(x_return_mean_rollout, _RETURN_MOMENT_STEP)
        head_value._x_return_variance.lerp_(x_return_variance_rollout, _RETURN_MOMENT_STEP)

        # W ← W·σ/σ', b ← (σ·b + μ − μ')/σ': under σ' and μ', every output is unchanged
        x_value_scale = _get_value_scale(head_value)
        layer.weight.mul_(x_value_scale_old / x_value_scale)
        layer.bias.copy_(
            (x_value_scale_old * layer.bias + x_return_mean_old - head_value._x_return_mean)
            / x_value_scale
        )

    return x_value_scale


def _get_value_scale(head_value: HeadValue) -> torch.Tensor:
    """Derive the value head's scale σ from its statistics.

    σ² is the mean over the streams of their running variance, floored at `_MIN_RETURN_VARIANCE`.

    Parameters
    ----------
    head_value : HeadValue
        The value head.

    Returns
    -------
    torch.Tensor
        (). σ, which every stream shares.
    """
    return head_value._x_return_variance.mean().clamp(min=_MIN_RETURN_VARIANCE).sqrt()
