import math

import torch
import torch.nn as nn
from slai import ACTION_SPEC_REGISTRY

from src.rl.batches import get_entity_kind_rows
from src.rl.constants import ACTION_TYPES
from src.rl.constants import NUM_ACTION_TYPES
from src.rl.types import CoreOutput
from src.rl.types import EntityKind


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
    pointer heads, so the same Entity maps to the same key in every selection context.

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
        """Score Entities by scaled key·query, each key against its query.

        Any identity dedup is baked into the rows the caller hands in (src/rl/encoding/mask.py);
        sampling is the caller's job.

        Parameters
        ----------
        x_keys : torch.Tensor
            (..., dim_key). The keys, from PointerKeys.
        x_global : torch.Tensor
            (Q, dim_global). Each query's global vector, one row per query: a state for the target
            head, a selection for the selection head.
        x_cond : torch.Tensor
            (Q, dim_cond). Each query's conditioning vector, which it reads with `x_global`.
        x_key_state : torch.Tensor
            (...) long. Each key's query row, broadcast to the keys' leading dims.
        x_mask : torch.Tensor
            (...) bool. The legal keys.

        Returns
        -------
        torch.Tensor
            (...). The masked logits: -inf where `x_mask` is False.
        """
        x_query = self._query_net(torch.cat([x_global, x_cond], dim=-1))  # (Q, dim_key)
        x_logit = (x_keys * x_query[x_key_state]).sum(-1) * self._scale

        return x_logit.masked_fill(~x_mask, float("-inf"))


class HeadSelection(nn.Module):
    """Selection head: scores each legal selection, a token under an `ActionType`.

    A selection's logit joins its `ActionType`'s in L1, the joint level.

    Parameters
    ----------
    dim_global : int
        The width of the global vector.
    dim_op : int
        The width of the `ActionType` embedding.
    dim_key : int
        The width of the pointer keys and the query.
    """

    def __init__(self, dim_global: int, dim_op: int, dim_key: int) -> None:
        super().__init__()

        # `ActionType` embedding for the L1 pointer selection
        self._action_type_emb = nn.Embedding(NUM_ACTION_TYPES, dim_op)
        self._pointer_logits = PointerLogits(dim_global, dim_op, dim_key)

    def init_logit_weights(self) -> None:
        """Initialize the logit layer near zero after the general linear init."""
        self._pointer_logits.init_logit_weights()

    def forward(
        self,
        core_output: CoreOutput,
        x_token_key: torch.Tensor,
        x_selection_state: torch.Tensor,
        x_selection_token_row: torch.Tensor,
        x_selection_action_type: torch.Tensor,
    ) -> torch.Tensor:
        """Score each legal selection: its token as the selection of its `ActionType`.

        Parameters
        ----------
        core_output : CoreOutput
            The core's output.
        x_token_key : torch.Tensor
            (N, dim_key). Each token's key, from PointerKeys.
        x_selection_state : torch.Tensor
            (P,) long. Each legal selection's state.
        x_selection_token_row : torch.Tensor
            (P,) long. Each legal selection's token, by its row among the tokens.
        x_selection_action_type : torch.Tensor
            (P,) long. Each legal selection's action type.

        Returns
        -------
        torch.Tensor
            (P,). Each selection's logit; every selection is legal.
        """
        num_selections = x_selection_token_row.shape[0]

        # Calculate the selections' logits
        return self._pointer_logits(
            x_token_key[x_selection_token_row],
            core_output.x_global[x_selection_state],
            self._action_type_emb(x_selection_action_type),
            torch.arange(num_selections, device=x_selection_token_row.device),
            torch.ones(num_selections, dtype=torch.bool, device=x_selection_token_row.device),
        )


class HeadTarget(nn.Module):
    """L2 target head: scores each Monster of a state as its selected token's target.

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

        # Buffer for each `ActionType`'s L2 requirement
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
            none: the L1 selection.

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
        x_mask_l2 = torch.cat([x_has_target[x_monster_state] & x_target_legal, ~x_has_target]) # (M + B,)

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
            x_mask_l2,
        )


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
