import itertools
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from slai import ACTION_SPEC_REGISTRY

from src.rl.constants import ACTION_TYPES
from src.rl.constants import NUM_ACTION_TYPES
from src.rl.encoding import ENCODING_DIM_FOCUS
from src.rl.encoding import focus_view
from src.rl.models.heads import HeadActionType
from src.rl.models.heads import HeadPointerSelect
from src.rl.models.heads import PointerKeys
from src.rl.types import CoreOutput
from src.rl.types import EntityKind
from src.rl.types import PolicyOutput
from src.rl.types import StateBatch
from src.rl.types import TokenKind


# Barrier probability floor
_KL_UNIFORM_LOG_P_FLOOR = math.log(1e-4)


@dataclass
class _GroupedRows:
    """The window's rows grouped by Entity kind, each Entity kind's rows state after state.

    The key nets split the refined rows by it and both pointer levels score in its order. A
    selection maps back to the window through `x_order`.
    """

    x_order: torch.Tensor  # (N,) long, each grouped row's row in the window
    x_state: torch.Tensor  # (N,) long, each grouped row's state
    x_rows_of_entity_kind: tuple[torch.Tensor, ...]  # Each Entity kind's (n, dim_entity) rows
    entity_counts: list[int]  # Each Entity kind's number of rows, indexed by `EntityKind`


class PolicyHeads(nn.Module):
    """The policy's three levels: an action type, the token it selects and its Monster target.

    The levels run in order, each given the selections before it. Given recorded actions, each
    level scores the recorded selection instead of making one, through the same computation.

    Parameters
    ----------
    dim_entity : int
        The width of the core's refined token rows.
    dim_global : int
        The width of the core's global vector.
    dim_ff_primary : int
        The width of the action-type head's hidden layer.
    dim_op : int
        The width of the `ActionType` embedding.
    dim_key : int
        The width of the pointer keys and queries.
    """

    _x_selects_token: torch.Tensor
    _x_selects_target: torch.Tensor

    def __init__(
        self, dim_entity: int, dim_global: int, dim_ff_primary: int, dim_op: int, dim_key: int
    ) -> None:
        super().__init__()

        self._dim_key = dim_key

        self._head_action_type = HeadActionType(
            dim_global,
            dim_ff_primary,
            num_choices=NUM_ACTION_TYPES,
            dim_context=ENCODING_DIM_FOCUS,
        )
        self._head_level_2 = HeadPointerSelect(dim_global, dim_op, dim_key)
        self._head_level_3 = HeadPointerSelect(dim_global, dim_entity, dim_key)

        # `ActionType` embedding for the L2 / L3 pointer selection
        self._action_type_embedding = nn.Embedding(NUM_ACTION_TYPES, dim_op)

        # One key net per selectable `EntityKind` (all but the Character)
        self._pointer_key_card = PointerKeys(dim_entity, dim_key)
        self._pointer_key_relic = PointerKeys(dim_entity, dim_key)
        self._pointer_key_potion = PointerKeys(dim_entity, dim_key)
        self._pointer_key_event = PointerKeys(dim_entity, dim_key)
        self._pointer_key_room = PointerKeys(dim_entity, dim_key)
        self._pointer_key_monster = PointerKeys(dim_entity, dim_key)

        # Buffers for each `ActionType`'s L2 / L3 requirements
        action_type_arities = [ACTION_SPEC_REGISTRY[action_type].arity for action_type in ACTION_TYPES]
        self.register_buffer(
            "_x_selects_token",
            torch.tensor([arity != (0, 0) for arity in action_type_arities]),
            persistent=False,
        )
        self.register_buffer(
            "_x_selects_target", torch.tensor([arity[1] == 2 for arity in action_type_arities]), persistent=False
        )

    def init_logit_weights(self) -> None:
        """Initialize each level's logit layer near zero after the general linear init."""
        nn.init.orthogonal_(self._head_action_type._scorer[-1].weight, gain=0.01)
        nn.init.orthogonal_(self._head_level_2._query_net[-1].weight, gain=0.01)
        nn.init.orthogonal_(self._head_level_3._query_net[-1].weight, gain=0.01)

    def forward(
        self,
        state_batch: StateBatch,
        core_output: CoreOutput,
        entity_counts: list[int],
        x_recorded_actions: torch.Tensor | None = None,
        greedy: bool = False,
        generator: torch.Generator | None = None,
    ) -> PolicyOutput:
        """Select each state's action, or score the recorded ones.

        L1 takes the action type, L2 the token it selects among its state's rows, and L3 the
        Monster that selection targets.

        Parameters
        ----------
        state_batch : StateBatch
            The game states.
        core_output : CoreOutput
            The core's output on `state_batch`.
        entity_counts : list[int]
            Each Entity kind's number of tokens, indexed by `EntityKind`.
        x_recorded_actions : torch.Tensor | None, default None
            (B, 3) long. Recorded action type, state-local token offset, and Monster target rank;
            None to select. Absent selections and targets hold -1.
        greedy : bool, default False
            Whether each level takes its argmax rather than a sample.
        generator : torch.Generator | None, default None
            The generator samples draw from, or None for the global one.

        Returns
        -------
        PolicyOutput
            Each state's action, selected or recorded, and each level's statistics.
        """
        refined = core_output.tokens
        x_legal_action_type = state_batch.x_legal_action_type

        # The recorded actions' columns, or None when selecting
        x_recorded_action_type, x_recorded_token_offset, x_recorded_target = (
            x_recorded_actions.unbind(-1) if x_recorded_actions is not None else (None,) * 3
        )

        # Select the `ActionType` for every row — one masked categorical, focus-gated.
        x_logits = self._head_action_type(
            core_output.x_policy_context, focus_view(state_batch.x_context), x_legal_action_type
        )
        x_action_type, x_log_prob_l1, x_entropy_l1, x_kl_uniform_l1 = _categorical_step(
            x_logits, x_legal_action_type, x_recorded_action_type, greedy, generator
        )

        # Each token's state
        x_state_window = torch.repeat_interleave(
            refined.x_lengths, output_size=refined.rows.shape[0]
        )  # (N,)

        # The rows grouped by Entity kind, Entity kind after Entity kind and each Entity kind's
        # state after state: what the key nets split and the scoring runs on. Selections map back
        # to the window through it
        x_order = torch.argsort(state_batch.tokens.rows.x_entity_kind, stable=True)  # (N,)
        x_state = x_state_window[x_order]

        # The split hands each class its rows
        x_rows_of_entity_kind = refined.rows[x_order].split(entity_counts)
        grouped_rows = _GroupedRows(x_order, x_state, x_rows_of_entity_kind, entity_counts)

        # Compute the `Action`'s selections from the chosen `ActionType`
        (
            x_token_selected,
            x_token_offset,
            x_log_prob_l2,
            x_entropy_l2,
            x_kl_uniform_l2,
        ) = self._select_token(
            state_batch,
            core_output,
            grouped_rows,
            x_action_type,
            x_recorded_token_offset,
            greedy,
            generator,
        )
        x_target, x_log_prob_l3, x_entropy_l3, x_kl_uniform_l3 = self._select_target(
            state_batch,
            core_output,
            grouped_rows,
            x_action_type,
            x_token_selected,
            x_recorded_target,
            greedy,
            generator,
        )

        # Stack per-level columns: contiguous, faster than strided writes to a preallocated buffer
        return PolicyOutput(
            x_actions=torch.stack([x_action_type, x_token_offset, x_target], dim=1),
            x_log_prob=torch.stack([x_log_prob_l1, x_log_prob_l2, x_log_prob_l3], dim=1),
            x_entropy=torch.stack([x_entropy_l1, x_entropy_l2, x_entropy_l3], dim=1),
            x_kl_uniform=torch.stack([x_kl_uniform_l1, x_kl_uniform_l2, x_kl_uniform_l3], dim=1),
        )

    def _select_token(
        self,
        state_batch: StateBatch,
        core_output: CoreOutput,
        grouped_rows: _GroupedRows,
        x_action_type: torch.Tensor,
        x_recorded_token_offset: torch.Tensor | None,
        greedy: bool,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Make each state's L2 selection, or score the recorded one.

        L2 selects a token among its state's rows, for the action types that select one.

        Parameters
        ----------
        state_batch : StateBatch
            The game states.
        core_output : CoreOutput
            The core's output on `state_batch`.
        grouped_rows : _GroupedRows
            The window's rows grouped by Entity kind.
        x_action_type : torch.Tensor
            (B,) long. Each state's action type, the L1 selection.
        x_recorded_token_offset : torch.Tensor | None
            (B,) long. Each recorded selection's state-local token offset, -1 where absent; None
            to select.
        greedy : bool
            Whether to take each state's highest logit rather than a sample.
        generator : torch.Generator | None
            The generator samples draw from, or None for the global one.

        Returns
        -------
        x_token_selected : torch.Tensor
            (B,) long. Each state's selected row in the window; its Character's where its action
            type selects no token.
        x_token_offset : torch.Tensor
            (B,) long. The selected token's offset in its state, -1 where the level doesn't apply.
        x_log_prob : torch.Tensor
            (B,). The selection's log-prob, 0 where the level doesn't apply.
        x_entropy : torch.Tensor
            (B,). The entropy, 0 where the level doesn't apply.
        x_kl_uniform : torch.Tensor
            (B,). KL(uniform‖π) over the legal tokens, 0 where the level doesn't apply.
        """
        refined = core_output.tokens
        x_order = grouped_rows.x_order
        x_state = grouped_rows.x_state
        batch_size = x_action_type.shape[0]
        x_states = torch.arange(batch_size, device=x_action_type.device)

        # Each state's first row in the window
        x_state_row = refined.x_lengths.cumsum(0) - refined.x_lengths  # (B,)

        # The rows before each Entity kind's among the grouped rows
        num_rows_before_entity_kind = [0, *itertools.accumulate(grouped_rows.entity_counts)]

        x_keys = self._compute_token_keys(grouped_rows)

        # L2: score every row against its state's query, conditioned on the `ActionType` embedding;
        # the selection mask keeps exactly the type's legal items
        x_selects_token = self._x_selects_token[x_action_type]
        x_mask_rows = state_batch.tokens.rows.x_legal_selection[x_order, x_action_type[x_state]]

        # Every state at once: a type that selects no token (TurnEnd...) gets its state's Character
        # row as a placeholder, so its categorical stays finite. One per state, past the earlier
        # Entity kinds' rows
        num_rows_before_character = num_rows_before_entity_kind[EntityKind.CHARACTER]
        x_mask_rows[num_rows_before_character + x_states] |= ~x_selects_token

        # Local offsets survive state reordering; add this batch's starts before grouping the rows.
        # An absent selection (-1) uses its state's Character placeholder.
        x_recorded_row = None
        if x_recorded_token_offset is not None:
            # Each window row's place among the grouped rows
            x_place = torch.empty_like(x_order)
            x_place[x_order] = torch.arange(x_order.shape[0], device=x_order.device)

            # The Characters' rows in the window: one per state among the grouped rows, in
            # state order
            x_character_row = x_order.narrow(0, num_rows_before_character, batch_size)
            x_recorded_token = torch.where(
                x_recorded_token_offset >= 0,
                x_state_row + x_recorded_token_offset,
                x_character_row,
            )
            x_recorded_row = x_place[x_recorded_token]

        # Calculate the rows' logits & select from each state's
        x_logits = self._head_level_2(
            x_keys,
            core_output.x_policy_context,
            self._action_type_embedding(x_action_type),
            x_state,
            x_mask_rows,
        )
        x_row_selected, x_log_prob, x_entropy, x_kl_uniform = _categorical_rows(
            x_logits, x_mask_rows, x_state, batch_size, x_recorded_row, greedy, generator
        )

        # The selected rows back in the window
        x_token_selected = x_order[x_row_selected]
        x_token_offset = x_token_selected - x_state_row

        # A level a row doesn't reach: no selection (-1) and zero terms, whatever its
        # placeholder drew
        return (
            x_token_selected,
            torch.where(x_selects_token, x_token_offset, -1),
            torch.where(x_selects_token, x_log_prob, 0.0),
            torch.where(x_selects_token, x_entropy, 0.0),
            torch.where(x_selects_token, x_kl_uniform, 0.0),
        )

    def _compute_token_keys(self, grouped_rows: _GroupedRows) -> torch.Tensor:
        """Project each grouped row to its L2 key.

        Parameters
        ----------
        grouped_rows : _GroupedRows
            The window's rows grouped by Entity kind.

        Returns
        -------
        torch.Tensor
            (N, dim_key). Each grouped row's key; zeros for the Monsters and the Characters.
        """
        x_rows_of_entity_kind = grouped_rows.x_rows_of_entity_kind
        entity_counts = grouped_rows.entity_counts
        x_rows_character = x_rows_of_entity_kind[EntityKind.CHARACTER]

        # L2 keys, once per token: its own class's net, the one every action type scores
        # it with. Monsters and the Character are never L2 selections: zero keys
        x_keys_of_entity_kind = {
            EntityKind.CARD: self._pointer_key_card(x_rows_of_entity_kind[EntityKind.CARD]),
            EntityKind.CHARACTER: torch.zeros(
                entity_counts[EntityKind.CHARACTER],
                self._dim_key,
                dtype=x_rows_character.dtype,
                device=x_rows_character.device,
            ),
            EntityKind.EVENT_OPTION: self._pointer_key_event(
                x_rows_of_entity_kind[EntityKind.EVENT_OPTION]
            ),
            EntityKind.MONSTER: torch.zeros(
                entity_counts[EntityKind.MONSTER],
                self._dim_key,
                dtype=x_rows_character.dtype,
                device=x_rows_character.device,
            ),
            EntityKind.POTION: self._pointer_key_potion(x_rows_of_entity_kind[EntityKind.POTION]),
            EntityKind.RELIC: self._pointer_key_relic(x_rows_of_entity_kind[EntityKind.RELIC]),
            EntityKind.ROOM: self._pointer_key_room(x_rows_of_entity_kind[EntityKind.ROOM]),
        }

        # The keys in the rows' order. Sorted keys, not `EntityKind` itself: compiled, an Enum
        # can't be iterated
        return torch.cat(
            [x_keys_of_entity_kind[entity_kind] for entity_kind in sorted(x_keys_of_entity_kind)]
        )

    def _select_target(
        self,
        state_batch: StateBatch,
        core_output: CoreOutput,
        grouped_rows: _GroupedRows,
        x_action_type: torch.Tensor,
        x_token_selected: torch.Tensor,
        x_recorded_target: torch.Tensor | None,
        greedy: bool,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Make each state's L3 selection, or score the recorded one.

        L3 selects a Monster among its state's, for the selections that take a target.

        Parameters
        ----------
        state_batch : StateBatch
            The game states.
        core_output : CoreOutput
            The core's output on `state_batch`.
        grouped_rows : _GroupedRows
            The window's rows grouped by Entity kind.
        x_action_type : torch.Tensor
            (B,) long. Each state's action type, the L1 selection.
        x_token_selected : torch.Tensor
            (B,) long. Each state's selected row in the window, the L2 selection.
        x_recorded_target : torch.Tensor | None
            (B,) long. Each recorded target's Monster rank, -1 where absent; None to select.
        greedy : bool
            Whether to take each state's highest logit rather than a sample.
        generator : torch.Generator | None
            The generator samples draw from, or None for the global one.

        Returns
        -------
        x_target : torch.Tensor
            (B,) long. The targeted Monster's rank in its state, -1 where the level doesn't apply.
        x_log_prob : torch.Tensor
            (B,). The selection's log-prob, 0 where the level doesn't apply.
        x_entropy : torch.Tensor
            (B,). The entropy, 0 where the level doesn't apply.
        x_kl_uniform : torch.Tensor
            (B,). KL(uniform‖π) over the legal targets, 0 where the level doesn't apply.
        """
        tokens = state_batch.tokens.rows
        batch_size = x_action_type.shape[0]
        x_states = torch.arange(batch_size, device=x_action_type.device)

        # The Monsters among the grouped rows
        num_monsters = grouped_rows.entity_counts[EntityKind.MONSTER]
        num_rows_before_entity_kind = [0, *itertools.accumulate(grouped_rows.entity_counts)]
        num_rows_before_monster = num_rows_before_entity_kind[EntityKind.MONSTER]
        x_monster_tokens = grouped_rows.x_rows_of_entity_kind[EntityKind.MONSTER]

        # L3: Monster target, only for actions that take one (CardPlay / PotionUse)
        # Keep only rows whose committed L2 selection takes a target: at recompute
        # the recorded selection says so (-1 = none), at sample time the mask does
        x_has_target = (
            x_recorded_target >= 0
            if x_recorded_target is not None
            else self._x_selects_target[x_action_type] & tokens.x_needs_target[x_token_selected]
        )

        # The choices: each Monster row, scored against its state's selected Card/potion token and
        # legal where its state takes a target its action type can name; then each state's "no
        # target" choice, a zero key, legal where it takes none
        x_monster_rows = grouped_rows.x_order.narrow(0, num_rows_before_monster, num_monsters)
        x_monster_state = grouped_rows.x_state.narrow(0, num_rows_before_monster, num_monsters)
        x_choice_state = torch.cat([x_monster_state, x_states])
        x_target_legal = tokens.x_legal_target[x_monster_rows, x_action_type[x_monster_state]]
        x_mask_l3 = torch.cat([x_has_target[x_monster_state] & x_target_legal, ~x_has_target])

        # A cat, not F.pad: compiled, F.pad's backward fails on an empty Monster block
        x_keys = torch.cat(
            [
                self._pointer_key_monster(x_monster_tokens),
                torch.zeros(
                    batch_size,
                    self._dim_key,
                    dtype=x_monster_tokens.dtype,
                    device=x_monster_tokens.device,
                ),
            ]
        )

        # A target is a Monster's rank: its choice is its state's first Monster choice plus
        # the rank
        x_count_monster = state_batch.state_counts.x_token_kind_count[:, TokenKind.MONSTERS]
        x_first_monster = x_count_monster.cumsum(0) - x_count_monster
        x_recorded_choice = None
        if x_recorded_target is not None:
            x_recorded_choice = torch.where(
                x_has_target, x_first_monster + x_recorded_target, num_monsters + x_states
            )

        # Calculate the choices' logits & select from each state's
        x_logits = self._head_level_3(
            x_keys,
            core_output.x_policy_context,
            core_output.tokens.rows[x_token_selected],
            x_choice_state,
            x_mask_l3,
        )
        x_choice, x_log_prob, x_entropy, x_kl_uniform = _categorical_rows(
            x_logits, x_mask_l3, x_choice_state, batch_size, x_recorded_choice, greedy, generator
        )
        x_target = x_choice - x_first_monster

        # A level a row doesn't reach: no selection (-1) and zero terms, whatever its
        # placeholder drew
        return (
            torch.where(x_has_target, x_target, -1),
            torch.where(x_has_target, x_log_prob, 0.0),
            torch.where(x_has_target, x_entropy, 0.0),
            torch.where(x_has_target, x_kl_uniform, 0.0),
        )


def _categorical_step(
    x_logits: torch.Tensor,
    x_mask: torch.Tensor,
    x_recorded_idx: torch.Tensor | None,
    greedy: bool,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute each row's selection, its log-prob, the entropy and KL(uniform‖π).

    Parameters
    ----------
    x_logits : torch.Tensor
        (B, N). A head's logits, which the head already masked.
    x_mask : torch.Tensor
        (B, N) bool. The KL's legal set.
    x_recorded_idx : torch.Tensor | None
        (B,) long. The recorded selections to score (PPO recompute), or None to select.
    greedy : bool
        Whether to select each row's highest logit rather than sample.
    generator : torch.Generator | None
        The generator samples draw from, or None for the global one.

    Returns
    -------
    x_idx : torch.Tensor
        (B,) long. Each row's selection.
    x_log_prob : torch.Tensor
        (B,). Its log-prob.
    x_entropy : torch.Tensor
        (B,). The entropy.
    x_kl_uniform : torch.Tensor
        (B,). KL(uniform‖π) over the legal set.
    """
    # No argument checks: each one reads the logits back, making the CPU wait for the GPU
    dist = torch.distributions.Categorical(logits=x_logits, validate_args=False)

    # Determine which index to sample
    if x_recorded_idx is not None:
        # PPO recompute path
        x_idx = x_recorded_idx
    elif greedy:
        # Highest logit
        x_idx = torch.argmax(x_logits, dim=-1)
    else:
        # Sample from distribution
        x_idx = torch.multinomial(dist.probs, 1, True, generator=generator).squeeze(-1)

    # KL(uniform‖π) over the legal set, the anti-collapse barrier. −log p clamps at the floor:
    # unbounded, the barrier dominates the loss once a head nears determinism
    x_n_legal = x_mask.sum(-1).clamp(min=1).float()  # min=1: empty/singleton rows → KL 0
    x_logp = dist.logits.clamp(min=_KL_UNIFORM_LOG_P_FLOOR)
    x_kl_uniform = -x_logp.masked_fill(~x_mask, 0.0).sum(-1) / x_n_legal - x_n_legal.log()

    return x_idx, dist.log_prob(x_idx), dist.entropy(), x_kl_uniform


def _sum_by_state(x_values: torch.Tensor, x_state: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Sum each state's rows' `x_values`.

    Parameters
    ----------
    x_values : torch.Tensor
        (R,). Each row's value.
    x_state : torch.Tensor
        (R,) long. Each row's state.
    batch_size : int
        The number of states, S.

    Returns
    -------
    torch.Tensor
        (S,). Each state's sum of its rows' `x_values`.
    """
    return torch.zeros(batch_size, dtype=x_values.dtype, device=x_values.device).index_add_(
        0, x_state, x_values
    )


def _max_by_state(x_values: torch.Tensor, x_state: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Find each state's max of its rows' `x_values`.

    Parameters
    ----------
    x_values : torch.Tensor
        (R,). Each row's value.
    x_state : torch.Tensor
        (R,) long. Each row's state.
    batch_size : int
        The number of states, S.

    Returns
    -------
    torch.Tensor
        (S,). Each state's max of its rows' `x_values`.
    """
    return torch.full(
        (batch_size,), float("-inf"), dtype=x_values.dtype, device=x_values.device
    ).scatter_reduce_(0, x_state, x_values, reduce="amax")


def _argmax_by_state(
    x_values: torch.Tensor, x_state: torch.Tensor, batch_size: int
) -> torch.Tensor:
    """Find each state's row of its highest `x_values`, the first of a tie.

    Parameters
    ----------
    x_values : torch.Tensor
        (R,). Each row's value.
    x_state : torch.Tensor
        (R,) long. Each row's state.
    batch_size : int
        The number of states, S.

    Returns
    -------
    torch.Tensor
        (S,) long. Each state's row of its highest `x_values`, the first of a tie.
    """
    num_rows = x_values.shape[0]
    x_row = torch.arange(num_rows, device=x_values.device)
    x_is_max = x_values == _max_by_state(x_values, x_state, batch_size)[x_state]
    return torch.full(
        (batch_size,), num_rows, dtype=torch.long, device=x_values.device
    ).scatter_reduce_(0, x_state, x_row.masked_fill(~x_is_max, num_rows), reduce="amin")


def _categorical_rows(
    x_logits: torch.Tensor,
    x_mask: torch.Tensor,
    x_state: torch.Tensor,
    batch_size: int,
    x_recorded_row: torch.Tensor | None,
    greedy: bool,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select over packed rows as `_categorical_step` does, one row per state.

    Samples are Gumbel-max: one draw per row from `generator`, or the global one when it is None,
    then each state's argmax.

    Parameters
    ----------
    x_logits : torch.Tensor
        (R,). The rows' logits, which the head already masked.
    x_mask : torch.Tensor
        (R,) bool. The KL's legal set.
    x_state : torch.Tensor
        (R,) long. Each row's state.
    batch_size : int
        The number of states, S.
    x_recorded_row : torch.Tensor | None
        (S,) long. The recorded selected rows to score (PPO recompute), or None to select.
    greedy : bool
        Whether to select each state's highest logit rather than sample.
    generator : torch.Generator | None
        The generator samples draw from, or None for the global one.

    Returns
    -------
    x_row : torch.Tensor
        (S,) long. Each state's selected row.
    x_log_prob : torch.Tensor
        (S,). Its log-prob.
    x_entropy : torch.Tensor
        (S,). The entropy.
    x_kl_uniform : torch.Tensor
        (S,). KL(uniform‖π) over the legal set.
    """
    # Each row's log-prob within its state; the max keeps the exponentials finite, and takes no
    # gradient, as in logsumexp
    x_shifted = x_logits - _max_by_state(x_logits.detach(), x_state, batch_size)[x_state]
    x_log_prob = x_shifted - _sum_by_state(x_shifted.exp(), x_state, batch_size).log()[x_state]

    # Determine which row to select
    if x_recorded_row is not None:
        # PPO recompute path
        x_row = x_recorded_row
    elif greedy:
        # Highest logit
        x_row = _argmax_by_state(x_logits, x_state, batch_size)
    else:
        # Sample: Gumbel noise per row, masked after it is added, then each state's highest
        x_gumbel = -torch.empty_like(x_logits).exponential_(generator=generator).log()
        x_score = (x_logits + x_gumbel).masked_fill(~x_mask, float("-inf"))
        x_row = _argmax_by_state(x_score, x_state, batch_size)

    # The entropy over the legal set; a masked row's log-prob is -inf, so it sits out as 0
    x_log_prob_legal = x_log_prob.masked_fill(~x_mask, 0.0)
    x_entropy = -_sum_by_state(x_log_prob_legal.exp() * x_log_prob_legal, x_state, batch_size)

    # KL(uniform‖π) over the legal set, −log p clamped at the floor, as in `_categorical_step`;
    # min=1: empty/singleton states → KL 0
    x_n_legal = _sum_by_state(x_mask.to(x_logits.dtype), x_state, batch_size).clamp(min=1)
    x_log_prob_floor = x_log_prob.clamp(min=_KL_UNIFORM_LOG_P_FLOOR).masked_fill(~x_mask, 0.0)
    x_log_prob_sum = _sum_by_state(x_log_prob_floor, x_state, batch_size)
    x_kl_uniform = -x_log_prob_sum / x_n_legal - x_n_legal.log()
    return x_row, x_log_prob[x_row], x_entropy, x_kl_uniform
