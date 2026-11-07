import math
from collections.abc import Callable

import torch
import torch.nn as nn
from slai import ACTION_SPEC_REGISTRY

from src.rl.constants import NUM_ACTION_TYPES
from src.rl.encoding.context import focus_view
from src.rl.encoding.focus import ENCODING_DIM_FOCUS
from src.rl.encoding.state import COL_START_CHARACTER
from src.rl.encoding.state import COL_START_HEALTH_BLOCK
from src.rl.encoding.state import COL_START_MONSTER
from src.rl.encoding.state import COL_START_POTION
from src.rl.encoding.state import COL_START_PRICE
from src.rl.encoding.state import COL_START_RELIC
from src.rl.encoding.state import COL_START_STASIS_SLOT
from src.rl.encoding.state import COL_START_TAG
from src.rl.models.core import Core
from src.rl.models.heads import HeadActionType
from src.rl.models.heads import HeadPointerSelect
from src.rl.models.heads import HeadValue
from src.rl.models.heads import PointerKeys
from src.rl.models.map_encoder import message_passing
from src.rl.reward import RewardStream
from src.rl.types import ACTION_TYPE_BY_INT
from src.rl.types import CoreOutput
from src.rl.types import Jagged
from src.rl.types import PolicyOutput
from src.rl.types import StateBatch
from src.rl.types import TokenIndex
from src.rl.types import TokenKind


# Barrier probability floor: each legal action's −log p contribution to the
# uniform-KL barrier caps at −log(1e-4) ≈ 9.2 nats
_KL_UNIFORM_LOG_P_FLOOR = math.log(1e-4)


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


class ActorCritic(nn.Module):
    # The buffers __init__ registers, typed for mypy
    _x_selects_token: torch.Tensor
    _x_is_targeted: torch.Tensor

    def __init__(
        self,
        dim_entity: int,
        dim_global: int,
        transformer_dim_ff: int,
        transformer_num_heads: int,
        transformer_num_blocks: int,
        gnn_num_layers: int,
        map_encoder_dim: int,
        dim_ff_primary: int,
        dim_ff_value: int,
        dim_op: int,
        dim_key: int,
    ) -> None:
        super().__init__()

        self.core = Core(
            dim_entity=dim_entity,
            dim_global=dim_global,
            transformer_dim_ff=transformer_dim_ff,
            transformer_num_heads=transformer_num_heads,
            transformer_num_blocks=transformer_num_blocks,
            gnn_num_layers=gnn_num_layers,
            map_encoder_dim=map_encoder_dim,
        )

        # L1: one masked categorical over ActionType, focus-gated (GLU)
        self._head_action_type = HeadActionType(
            dim_global,
            dim_ff_primary,
            num_choices=NUM_ACTION_TYPES,
            dim_context=ENCODING_DIM_FOCUS,
        )
        self._head_level_2 = HeadPointerSelect(dim_global, dim_op, dim_key)
        self._head_level_3 = HeadPointerSelect(dim_global, dim_entity, dim_key)
        self._head_value = HeadValue(dim_global, dim_ff_value, len(RewardStream))

        # L2/L3 pointer selection: one key net per Entity class + a shared query net per level
        self._action_type_embedding = nn.Embedding(NUM_ACTION_TYPES, dim_op)

        # One key net per Entity class; every TokenKind of that class shares it, so an Entity maps
        # to the same key in every selection context
        self._pointer_key_card = PointerKeys(dim_entity, dim_key)
        self._pointer_key_relic = PointerKeys(dim_entity, dim_key)
        self._pointer_key_potion = PointerKeys(dim_entity, dim_key)
        self._pointer_key_event = PointerKeys(dim_entity, dim_key)
        self._pointer_key_room = PointerKeys(dim_entity, dim_key)
        self._pointer_key_monster = PointerKeys(dim_entity, dim_key)
        self._dim_key = dim_key

        # Per policy action type, by its engine arity: whether it makes an L2 selection, and
        # whether that selection can take an L3 target; derived, so excluded from checkpoints
        arities = [ACTION_SPEC_REGISTRY[action_type].arity for action_type in ACTION_TYPE_BY_INT]
        self.register_buffer(
            "_x_selects_token",
            torch.tensor([arity != (0, 0) for arity in arities]),
            persistent=False,
        )
        self.register_buffer(
            "_x_is_targeted", torch.tensor([arity[1] == 2 for arity in arities]), persistent=False
        )

        # The policy heads' computation as a plain function, which compile_hot_paths swaps for
        # its compiled version on the Learner's model
        self.policy_heads_recorded = _policy_heads

        self._init_weights()

    def _init_weights(self) -> None:
        """Apply the PPO init.

        Orthogonal (√2), near-zero logit layers (policy ≈ uniform), unit-gain value.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:  # Some projector layers are bias-free
                    nn.init.zeros_(m.bias)

        # The shared first layer per column range, each Entity kind's then the health/block, price,
        # slot and Tag ones, as its own layer: orthogonal over all of them would shrink each
        # column's scale
        with torch.no_grad():
            for x_weight_range in torch.tensor_split(
                self.core._entity_projector._pre_token.weight,
                (
                    COL_START_RELIC,
                    COL_START_POTION,
                    COL_START_MONSTER,
                    COL_START_CHARACTER,
                    COL_START_HEALTH_BLOCK,
                    COL_START_PRICE,
                    COL_START_STASIS_SLOT,
                    COL_START_TAG,
                ),
                dim=1,
            ):
                nn.init.orthogonal_(x_weight_range, gain=math.sqrt(2))
        nn.init.orthogonal_(self._head_action_type._scorer[-1].weight, gain=0.01)
        nn.init.orthogonal_(self._head_level_2._query_net[-1].weight, gain=0.01)
        nn.init.orthogonal_(self._head_level_3._query_net[-1].weight, gain=0.01)
        nn.init.orthogonal_(self._head_value._network[-1].weight, gain=1.0)

    def _run_policy_heads(
        self,
        heads: Callable[..., tuple[torch.Tensor, ...]],
        game_state: StateBatch,
        core_out: CoreOutput,
        greedy: bool,
        generator: torch.Generator | None = None,
        x_recorded_actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        return heads(
            self,
            x_focus=focus_view(game_state.x_context),
            x_global=core_out.x_policy_context,
            refined=core_out.tokens,
            index=core_out.index,
            x_legal_selection=game_state.tokens.rows.x_legal_selection,
            x_legal_target=game_state.tokens.rows.x_legal_target,
            x_needs_target=game_state.tokens.rows.x_needs_target,
            x_legal_action_type=game_state.x_legal_action_type,
            greedy=greedy,
            generator=generator,
            x_recorded_actions=x_recorded_actions,
        )

    def forward(self, game_state: StateBatch) -> torch.Tensor:
        """Compute the values.

        Parameters
        ----------
        game_state : StateBatch
            The game states.

        Returns
        -------
        torch.Tensor
            (B, len(RewardStream)). The values.
        """
        return self._head_value(self.core(game_state).x_value_context)

    def act(
        self,
        game_state: StateBatch,
        greedy: bool = False,
        generator: torch.Generator | None = None,
    ) -> PolicyOutput:
        """Select each state's action.

        Without values to compute, the core runs the draw-order layer only on the states the
        policy reads.

        Parameters
        ----------
        game_state : StateBatch
            The game states.
        greedy : bool, default False
            Whether each level takes its argmax rather than a sample.
        generator : torch.Generator | None, default None
            The generator samples draw from, or None for the global one.

        Returns
        -------
        PolicyOutput
            The selected actions and each level's statistics. Actions use the argmax under
            `greedy`, otherwise samples from `generator`.
        """
        core_out = self.core(game_state, with_value=False)

        # Sampling actions runs eager
        (
            x_action_type,
            x_token_offset,
            x_target,
            x_log_prob,
            x_entropy,
            x_kl_uniform,
        ) = self._run_policy_heads(_policy_heads, game_state, core_out, greedy, generator)
        return PolicyOutput(
            x_actions=torch.stack([x_action_type, x_token_offset, x_target], dim=1),
            x_log_prob=x_log_prob,
            x_entropy=x_entropy,
            x_kl_uniform=x_kl_uniform,
        )

    def evaluate_actions(
        self, game_state: StateBatch, x_recorded_actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score the recorded actions and compute the values.

        Parameters
        ----------
        game_state : StateBatch
            The game states.
        x_recorded_actions : torch.Tensor
            (B, 3) long. Each state's recorded action: action type, token offset within that
            state, Monster target rank; -1 where a selection or target didn't apply.

        Returns
        -------
        x_log_prob : torch.Tensor
            (B,). The recorded actions' log-prob, summed over levels.
        x_entropy : torch.Tensor
            (B, 3). Each level's entropy.
        x_kl_uniform : torch.Tensor
            (B, 3). Each level's KL(uniform‖π).
        x_values : torch.Tensor
            (B, len(RewardStream)). The values.
        """
        core_out = self.core(game_state)
        x_values = self._head_value(core_out.x_value_context)

        # Recorded actions run the attribute, compiled on the Learner
        _, _, _, x_log_prob, x_entropy, x_kl_uniform = self._run_policy_heads(
            self.policy_heads_recorded,
            game_state,
            core_out,
            greedy=True,
            x_recorded_actions=x_recorded_actions,
        )
        return x_log_prob.sum(-1), x_entropy, x_kl_uniform, x_values


def _policy_heads(
    model: ActorCritic,
    x_focus: torch.Tensor,
    x_global: torch.Tensor,
    refined: Jagged[torch.Tensor],
    index: TokenIndex,
    x_legal_selection: torch.Tensor,
    x_legal_target: torch.Tensor,
    x_needs_target: torch.Tensor,
    x_legal_action_type: torch.Tensor,
    greedy: bool,
    generator: torch.Generator | None,
    x_recorded_actions: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    """Make `model`'s L1-L3 selections, or score the recorded ones.

    The heads read the core's global vector and refined rows of the packed window, given the L1
    mask. L2 and L3 select per state over its rows, grouped by Entity kind.

    Parameters
    ----------
    model : ActorCritic
        The model whose heads run.
    x_focus : torch.Tensor
        (B, ENCODING_DIM_FOCUS) float32. Each state's focus row, the L1 head's context.
    x_global : torch.Tensor
        (B, dim_global). The core's global vector of each state.
    refined : Jagged[torch.Tensor]
        The core's (N, dim_entity) refined token rows, segmented by state.
    index : TokenIndex
        The token kind, state and row lookup tables.
    x_legal_selection : torch.Tensor
        (N, NUM_ACTION_TYPES) bool. Each token's legal selections.
    x_legal_target : torch.Tensor
        (N, NUM_ACTION_TYPES) bool. Each token's legal Monster targets.
    x_needs_target : torch.Tensor
        (N,) bool. Whether a token's selection needs a Monster target.
    x_legal_action_type : torch.Tensor
        (B, NUM_ACTION_TYPES) bool. The L1 mask.
    greedy : bool
        Whether each level takes its argmax rather than a sample.
    generator : torch.Generator | None
        The generator samples draw from, or None for the global one.
    x_recorded_actions : torch.Tensor | None
        (B, 3) long. Recorded action type, state-local token offset, and Monster target rank;
        None to select. Absent selections and targets hold -1.

    Returns
    -------
    tuple[torch.Tensor, ...]
        The action type, the L2 selection's state-local token offset, and the L3 Monster target
        rank, each (B,) long and -1 where its level doesn't apply; then each level's log-prob,
        entropy and KL(uniform‖π), each (B, 3) and 0 where the level doesn't apply.
    """
    # The recorded actions' columns, or None when selecting
    x_recorded_action_type, x_recorded_token_offset, x_recorded_target = (
        x_recorded_actions.unbind(-1) if x_recorded_actions is not None else (None,) * 3
    )

    # Select the `ActionType` for every row — one masked categorical, focus-gated.
    x_logits = model._head_action_type(x_global, x_focus, x_legal_action_type)
    x_action_type, x_log_prob_l1, x_entropy_l1, x_kl_uniform_l1 = _categorical_step(
        x_logits, x_legal_action_type, x_recorded_action_type, greedy, generator
    )

    # Compute the `Action`'s selections from the chosen `ActionType`
    batch_size = x_action_type.shape[0]
    x_states = torch.arange(batch_size, device=x_action_type.device)
    x_state_first = refined.x_lengths.cumsum(0) - refined.x_lengths

    # The rows grouped by Entity kind, Entity kind after Entity kind and each Entity kind's state
    # after state: what the key nets split and the scoring runs on. Selections map back to the
    # window through it
    x_order = torch.argsort(index.x_entity_kind, stable=True)  # (N,)
    x_state = index.x_state[x_order]
    (
        num_cards,
        num_relics,
        num_potions,
        num_monsters,
        num_event_options,
        num_character,
        _,
    ) = index.entity_counts

    # The split hands each class its rows
    x_cards, x_relics, x_potions, x_monster_tokens, x_event_options, _, x_rooms = refined.rows[
        x_order
    ].split(index.entity_counts)

    # L2 keys, once per token: its own class's net, the one every action type scores
    # it with. Monsters and the Character are never L2 selections: zero keys
    x_keys = torch.cat(
        [
            model._pointer_key_card(x_cards),
            model._pointer_key_relic(x_relics),
            model._pointer_key_potion(x_potions),
            torch.zeros(
                num_monsters,
                model._dim_key,
                dtype=refined.rows.dtype,
                device=refined.rows.device,
            ),
            model._pointer_key_event(x_event_options),
            torch.zeros(
                num_character,
                model._dim_key,
                dtype=refined.rows.dtype,
                device=refined.rows.device,
            ),
            model._pointer_key_room(x_rooms),
        ]
    )

    # L2: score every row against its state's query, conditioned on the `ActionType` embedding;
    # the selection mask keeps exactly the type's legal items
    x_selects_token = model._x_selects_token[x_action_type]
    x_legal_for_action = x_legal_selection.gather(1, x_action_type[index.x_state].unsqueeze(1))
    x_mask_rows = x_legal_for_action.squeeze(1)[x_order]

    num_rows_before_monster = num_cards + num_relics + num_potions

    # Every state at once: a type that selects no token (TurnEnd...) gets its state's Character
    # row as a placeholder, so its categorical stays finite. One per state, past the earlier
    # Entity kinds' rows
    num_rows_before_character = num_rows_before_monster + num_monsters + num_event_options
    x_mask_rows[num_rows_before_character + x_states] |= ~x_selects_token

    # Local offsets survive state reordering; add this batch's starts before grouping the rows.
    # An absent selection (-1) uses its state's Character placeholder.
    x_recorded_row = None
    if x_recorded_token_offset is not None:
        # Each window row's place among the grouped rows
        x_place = torch.empty_like(x_order)
        x_place[x_order] = torch.arange(x_order.shape[0], device=x_order.device)
        x_recorded_token = torch.where(
            x_recorded_token_offset >= 0,
            x_state_first + x_recorded_token_offset,
            index.x_kind_row[:, TokenKind.CHARACTER],
        )
        x_recorded_row = x_place[x_recorded_token]

    # Calculate the rows' logits & select from each state's
    x_logits = model._head_level_2(
        x_keys, x_global, model._action_type_embedding(x_action_type), x_state, x_mask_rows
    )
    x_row_selected, x_log_prob_l2, x_entropy_l2, x_kl_uniform_l2 = _categorical_rows(
        x_logits, x_mask_rows, x_state, batch_size, x_recorded_row, greedy, generator
    )

    # The selected rows back in the window
    x_token_selected = x_order[x_row_selected]
    x_token_offset = x_token_selected - x_state_first

    # L3: Monster target, only for actions that take one (CardPlay / PotionUse)
    # Keep only rows whose committed L2 selection takes a target: at recompute
    # the recorded selection says so (-1 = none), at sample time the mask does
    x_has_target = (
        x_recorded_target >= 0
        if x_recorded_target is not None
        else model._x_is_targeted[x_action_type] & x_needs_target[x_token_selected]
    )

    # The choices: each Monster row, scored against its state's selected Card/potion token and
    # legal where its state takes a target its action type can name; then each state's "no
    # target" choice, a zero key, legal where it takes none
    x_monster_rows = x_order.narrow(0, num_rows_before_monster, num_monsters)
    x_monster_state = x_state.narrow(0, num_rows_before_monster, num_monsters)
    x_choice_state = torch.cat([x_monster_state, x_states])
    x_target_legal = x_legal_target[x_monster_rows, x_action_type[x_monster_state]]
    x_mask_l3 = torch.cat([x_has_target[x_monster_state] & x_target_legal, ~x_has_target])

    # A cat, not F.pad: compiled, F.pad's backward fails on an empty Monster block
    x_keys = torch.cat(
        [
            model._pointer_key_monster(x_monster_tokens),
            torch.zeros(
                batch_size,
                model._dim_key,
                dtype=x_monster_tokens.dtype,
                device=x_monster_tokens.device,
            ),
        ]
    )

    # A target is a Monster's rank: its choice is its state's first Monster choice plus the rank
    x_count_monster = index.x_kind_count[:, TokenKind.MONSTERS]
    x_first_monster = x_count_monster.cumsum(0) - x_count_monster
    x_recorded_choice = None
    if x_recorded_target is not None:
        x_recorded_choice = torch.where(
            x_has_target, x_first_monster + x_recorded_target, num_monsters + x_states
        )

    # Calculate the choices' logits & select from each state's
    x_logits = model._head_level_3(
        x_keys, x_global, refined.rows[x_token_selected], x_choice_state, x_mask_l3
    )
    x_choice, x_log_prob_l3, x_entropy_l3, x_kl_uniform_l3 = _categorical_rows(
        x_logits, x_mask_l3, x_choice_state, batch_size, x_recorded_choice, greedy, generator
    )
    x_target = x_choice - x_first_monster

    # A level a row doesn't reach: no selection (-1) and zero terms, whatever its placeholder drew
    x_token_offset = torch.where(x_selects_token, x_token_offset, -1)
    x_log_prob_l2 = torch.where(x_selects_token, x_log_prob_l2, 0.0)
    x_entropy_l2 = torch.where(x_selects_token, x_entropy_l2, 0.0)
    x_kl_uniform_l2 = torch.where(x_selects_token, x_kl_uniform_l2, 0.0)
    x_target = torch.where(x_has_target, x_target, -1)
    x_log_prob_l3 = torch.where(x_has_target, x_log_prob_l3, 0.0)
    x_entropy_l3 = torch.where(x_has_target, x_entropy_l3, 0.0)
    x_kl_uniform_l3 = torch.where(x_has_target, x_kl_uniform_l3, 0.0)

    # Stack per-level columns: contiguous, faster than strided writes to a preallocated buffer
    return (
        x_action_type,
        x_token_offset,
        x_target,
        torch.stack([x_log_prob_l1, x_log_prob_l2, x_log_prob_l3], dim=1),
        torch.stack([x_entropy_l1, x_entropy_l2, x_entropy_l3], dim=1),
        torch.stack([x_kl_uniform_l1, x_kl_uniform_l2, x_kl_uniform_l3], dim=1),
    )


def compile_hot_paths(model: ActorCritic) -> None:
    """Fuse the update's hot paths.

    They are each transformer block, the draw-order layer's block, the Map GNN's message passing,
    and the policy heads on recorded actions. All but the heads are autotuned, so the first update
    compiles for a minute or two.

    Parameters
    ----------
    model : ActorCritic
        The model, compiled in place.
    """
    mode = "max-autotune-no-cudagraphs"
    for block in model.core._entity_transformer._entity_transformer_blocks:
        block.compile(dynamic=True, mode=mode)
    model.core._draw_order_layer._draw_order_block.compile(dynamic=True, mode=mode)
    model.core._map_gnn.message_passing = torch.compile(message_passing, dynamic=True, mode=mode)

    # Not autotuned: an empty block's key-net matmul fails to lower under autotuning
    model.policy_heads_recorded = torch.compile(_policy_heads, dynamic=True)
