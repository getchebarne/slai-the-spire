import math

import torch
import torch.nn as nn

from src.rl.batches import get_entity_kind_rows
from src.rl.batches import get_owner_first_row
from src.rl.batches import get_row_owner
from src.rl.models.core import Core
from src.rl.models.heads import HeadActionType
from src.rl.models.heads import HeadTarget
from src.rl.models.heads import HeadSelection
from src.rl.models.heads import HeadValue
from src.rl.models.heads import PointerKeys
from src.rl.models.map_encoder import message_passing
from src.rl.reward import RewardStream
from src.rl.types import CoreOutput
from src.rl.types import EntityKind
from src.rl.types import Live
from src.rl.types import MapCache
from src.rl.types import PolicyOutput
from src.rl.types import States
from src.rl.types import TokenKind


# Barrier probability floor
_KL_UNIFORM_LOG_P_FLOOR = math.log(1e-4)


class ActorCritic(nn.Module):
    """The policy and the value function over a batch of game states.

    The policy has three levels: an action type, the token it selects and its Monster target. The
    levels run in order, each given the selections before it, and each has a head that returns
    its logits.

    Each entry point runs the core once, then the heads its caller needs:
    - `action_select`: the Actor's step and the eval's. Selects each state's action, without
      values.
    - `action_evaluate`: the Learner's update. Scores recorded actions and computes the values.
    - `get_values`: the Learner's bootstrap. Computes the values alone.
    """

    def __init__(
        self,
        dim_entity: int,
        dim_global: int,
        transformer_dim_ff: int,
        transformer_num_heads: int,
        transformer_num_blocks: int,
        gnn_num_layers: int,
        dim_map: int,
        dim_ff_primary: int,
        dim_ff_value: int,
        dim_op: int,
        dim_key: int,
    ) -> None:
        super().__init__()

        # Core
        self.core = Core(
            dim_entity=dim_entity,
            dim_global=dim_global,
            transformer_dim_ff=transformer_dim_ff,
            transformer_num_heads=transformer_num_heads,
            transformer_num_blocks=transformer_num_blocks,
            gnn_num_layers=gnn_num_layers,
            dim_map=dim_map,
        )

        # Policy heads, one per level
        self._head_action_type = HeadActionType(dim_global, dim_ff_primary)
        self._head_selection = HeadSelection(dim_global, dim_op, dim_key)
        self._head_target = HeadTarget(dim_global, dim_entity, dim_key)

        # One key net for every token, shared by the L2 and L3 pointer selections
        self._pointer_keys = PointerKeys(dim_entity, dim_key)

        # Value head
        self._head_value = HeadValue(dim_global, dim_ff_value, len(RewardStream))

        # The selections' statistics function, which compile_hot_paths swaps for its compiled self
        self._evaluate_selections = _evaluate_selections

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply the PPO init.

        Orthogonal (sqrt(2)), near-zero logit layers (policy ≈ uniform), unit-gain value.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.core.init_token_weights()
        self._head_action_type.init_logit_weights()
        self._head_selection.init_logit_weights()
        self._head_target.init_logit_weights()
        nn.init.orthogonal_(self._head_value._network[-1].weight, gain=1.0)

    def action_select(
        self,
        states: States[Live],
        greedy: bool = False,
        generator: torch.Generator | None = None,
        map_cache: MapCache | None = None,
    ) -> PolicyOutput:
        """Select each state's action.

        Each level selects from its head's logits.

        Parameters
        ----------
        states : States[Live]
            The game states.
        greedy : bool, default False
            Whether each level takes its argmax rather than a sample.
        generator : torch.Generator | None, default None
            The generator samples draw from, or None for the global one.
        map_cache : MapCache | None, default None
            The Map GNN's outputs to reuse and update within a rollout (see `MapGNN.forward`).

        Returns
        -------
        PolicyOutput
            The selected actions and each level's statistics. Actions use the argmax under
            `greedy`, otherwise samples from `generator`.
        """
        core_output = self.core(states, map_cache=map_cache)
        tokens_refined = core_output.tokens
        batch_size = tokens_refined.x_count.shape[0]

        # Select the `ActionType` for every row — one masked categorical
        x_logit_action_type = self._head_action_type(core_output)
        x_action_type = _select_idx(x_logit_action_type, greedy, generator)

        # Compute the `Action`'s selections from the chosen `ActionType`
        x_token_key = self._pointer_keys(tokens_refined.features.x_value)  # (N, dim_key)
        x_logit_selection = self._head_selection(core_output, x_token_key, x_action_type)
        x_selection_row = _select_row(
            x_logit_selection, core_output.x_token_state, batch_size, greedy, generator
        )

        # Each target row's state, the Monsters counted on the host
        num_monsters = int(states.x_total_cpu[:, TokenKind.MONSTERS].sum())
        x_target_state = _get_target_state(core_output, num_monsters)  # (M + B,)
        x_logit_target = self._head_target(
            core_output, x_token_key, x_target_state, x_action_type, x_selection_row
        )
        x_target_row = _select_row(x_logit_target, x_target_state, batch_size, greedy, generator)

        return self._evaluate_selections(
            core_output,
            x_logit_action_type,
            x_action_type,
            x_logit_selection,
            x_selection_row,
            x_logit_target,
            x_target_state,
            x_target_row,
        )

    def action_evaluate(
        self,
        states: States[Live],
        x_recorded_action: torch.Tensor,
    ) -> tuple[PolicyOutput, torch.Tensor]:
        """Score the recorded actions and compute the values.

        Each level's head scores its choices given the recorded selections before it.

        Parameters
        ----------
        states : States[Live]
            The game states.
        x_recorded_action : torch.Tensor
            (B, 3) long. Each state's recorded action: action type, selection offset within that
            state, Monster target rank; -1 where a selection or target didn't apply.

        Returns
        -------
        policy_output : PolicyOutput
            The recorded actions and each level's statistics.
        x_value : torch.Tensor
            (B, len(RewardStream)). The values.
        """
        core_output = self.core(states)
        x_value = self._head_value(core_output.x_global)
        tokens_refined = core_output.tokens

        # Pointer keys
        x_token_key = self._pointer_keys(tokens_refined.features.x_value)  # (N, dim_key)

        # Each target row's state, the Monsters counted on the host
        num_monsters = int(states.x_total_cpu[:, TokenKind.MONSTERS].sum())
        x_target_state = _get_target_state(core_output, num_monsters)  # (M + B,)

        # The recorded selections, each as a row of its level's logits
        x_action_type, x_selection_off, x_target = x_recorded_action.unbind(-1)
        x_selection_row = _row_of_selection_off(core_output, x_selection_off)
        x_target_row = _row_of_target(core_output, num_monsters, x_target)

        # Each level's logits, given the recorded selections before it
        x_logit_action_type = self._head_action_type(core_output)
        x_logit_selection = self._head_selection(core_output, x_token_key, x_action_type)
        x_logit_target = self._head_target(
            core_output, x_token_key, x_target_state, x_action_type, x_selection_row
        )

        policy_output = self._evaluate_selections(
            core_output,
            x_logit_action_type,
            x_action_type,
            x_logit_selection,
            x_selection_row,
            x_logit_target,
            x_target_state,
            x_target_row,
        )

        return policy_output, x_value

    def get_values(self, states: States[Live]) -> torch.Tensor:
        """Compute the values.

        Parameters
        ----------
        states : States[Live]
            The game states.

        Returns
        -------
        torch.Tensor
            (B, len(RewardStream)). The values.
        """
        return self._head_value(self.core(states).x_global)


def compile_hot_paths(model: ActorCritic) -> None:
    """Fuse the update's hot paths.

    They are each transformer block, the Map GNN's message passing, the key net, each policy head
    and the selections' statistics. The first two are autotuned, so the first update compiles for
    a minute or two.

    Parameters
    ----------
    model : ActorCritic
        The model, compiled in place.
    """
    mode = "max-autotune-no-cudagraphs"
    for block in model.core._entity_transformer._entity_transformer_blocks:
        block.compile(dynamic=True, mode=mode)
    model.core._map_gnn.message_passing = torch.compile(message_passing, dynamic=True, mode=mode)

    # Not autotuned: an empty block's key-net matmul fails to lower under autotuning
    model._pointer_keys.compile(dynamic=True)
    model._head_action_type.compile(dynamic=True)
    model._head_selection.compile(dynamic=True)
    model._head_target.compile(dynamic=True)
    model._evaluate_selections = torch.compile(_evaluate_selections, dynamic=True)


def _get_target_state(core_output: CoreOutput, num_monsters: int) -> torch.Tensor:
    """Find each target row's state.

    The target rows are the Monsters, state after state, then each state's "no target".

    Parameters
    ----------
    core_output : CoreOutput
        The core's output.
    num_monsters : int
        The batch's number of Monster tokens, counted on the host.

    Returns
    -------
    torch.Tensor
        (M + B,) long. Each target row's state.
    """
    batch_size = core_output.tokens.x_count.shape[0]
    x_count_monster = core_output.x_total[:, TokenKind.MONSTERS]
    x_monster_state = get_row_owner(x_count_monster, num_monsters)  # (M,)
    x_idx_state = torch.arange(batch_size, device=x_monster_state.device)

    return torch.cat([x_monster_state, x_idx_state])


def _row_of_selection_off(core_output: CoreOutput, x_selection_off: torch.Tensor) -> torch.Tensor:
    """Find each state's token at `x_selection_off` among the batch's tokens.

    Parameters
    ----------
    core_output : CoreOutput
        The core's output.
    x_selection_off : torch.Tensor
        (B,) long. Each state's selection offset in its state, -1 where none.

    Returns
    -------
    torch.Tensor
        (B,) long. Each state's token's row among the batch's tokens; its Character's where the
        offset is -1.
    """
    batch_size = x_selection_off.shape[0]
    x_state_first_token = get_owner_first_row(core_output.tokens.x_count)  # (B,)
    x_character_row = get_entity_kind_rows(
        core_output.tokens.x_entity_kind, EntityKind.CHARACTER, batch_size
    )  # (B,)

    return torch.where(
        x_selection_off >= 0, x_state_first_token + x_selection_off, x_character_row
    )


def _row_of_target(
    core_output: CoreOutput, num_monsters: int, x_target: torch.Tensor
) -> torch.Tensor:
    """Find each state's target `x_target` among the batch's target rows.

    Parameters
    ----------
    core_output : CoreOutput
        The core's output.
    num_monsters : int
        The batch's number of Monster tokens, counted on the host.
    x_target : torch.Tensor
        (B,) long. Each state's targeted Monster's rank in its state, -1 where none.

    Returns
    -------
    torch.Tensor
        (B,) long. Each state's target's row among the batch's target rows; its "no target" row
        where the rank is -1.
    """
    x_idx_state = torch.arange(x_target.shape[0], device=x_target.device)

    # A target is a Monster's rank: its row is its state's first Monster row plus the rank
    x_count_monster = core_output.x_total[:, TokenKind.MONSTERS]
    x_state_first_monster = get_owner_first_row(x_count_monster)  # (B,)

    return torch.where(x_target >= 0, x_state_first_monster + x_target, num_monsters + x_idx_state)


def _select_idx(
    x_logit: torch.Tensor, greedy: bool, generator: torch.Generator | None
) -> torch.Tensor:
    """Select each state's index among its logits.

    Parameters
    ----------
    x_logit : torch.Tensor
        (B, C). A head's logits, C per state, -inf where illegal.
    greedy : bool
        Whether to select each state's highest logit rather than sample.
    generator : torch.Generator | None
        The generator samples draw from, or None for the global one.

    Returns
    -------
    torch.Tensor
        (B,) long. Each state's selected index.
    """
    if greedy:
        # Highest logit
        return torch.argmax(x_logit, dim=-1)

    # No argument checks: each one reads the logits back, making the CPU wait for the GPU
    dist = torch.distributions.Categorical(logits=x_logit, validate_args=False)

    # Sample from distribution
    return torch.multinomial(dist.probs, 1, True, generator=generator).squeeze(-1)


def _select_row(
    x_logit: torch.Tensor,
    x_row_state: torch.Tensor,
    batch_size: int,
    greedy: bool,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Select each state's row among its rows of logits, as `_select_idx` does among indexes.

    Samples are Gumbel-max: one draw per row from `generator`, or the global one when it is None,
    then each state's argmax.

    Parameters
    ----------
    x_logit : torch.Tensor
        (R,). A head's logits, one per row, -inf where illegal.
    x_row_state : torch.Tensor
        (R,) long. Each row's state.
    batch_size : int
        The number of states, B.
    greedy : bool
        Whether to select each state's highest logit rather than sample.
    generator : torch.Generator | None
        The generator samples draw from, or None for the global one.

    Returns
    -------
    torch.Tensor
        (B,) long. Each state's selected row.
    """
    if greedy:
        # Highest logit
        return _argmax_by_state(x_logit, x_row_state, batch_size)

    # Sample: Gumbel noise per row, masked after it is added, then each state's highest
    x_gumbel = -torch.empty_like(x_logit).exponential_(generator=generator).log()
    x_score = (x_logit + x_gumbel).masked_fill(x_logit == float("-inf"), float("-inf"))

    return _argmax_by_state(x_score, x_row_state, batch_size)


def _evaluate_selections(
    core_output: CoreOutput,
    x_logit_action_type: torch.Tensor,
    x_action_type: torch.Tensor,
    x_logit_selection: torch.Tensor,
    x_selection_row: torch.Tensor,
    x_logit_target: torch.Tensor,
    x_target_state: torch.Tensor,
    x_target_row: torch.Tensor,
) -> PolicyOutput:
    """Compute each level's statistics of the selections, and the selections as actions.

    Parameters
    ----------
    core_output : CoreOutput
        The core's output.
    x_logit_action_type : torch.Tensor
        (B, NUM_ACTION_TYPES). The L1 logits.
    x_action_type : torch.Tensor
        (B,) long. Each state's action type, the L1 selection.
    x_logit_selection : torch.Tensor
        (N,). The L2 logits, one per token.
    x_selection_row : torch.Tensor
        (B,) long. Each state's selected token's row among the tokens, its Character's where
        none: the L2 selection.
    x_logit_target : torch.Tensor
        (M + B,). The L3 logits, one per target row.
    x_target_state : torch.Tensor
        (M + B,) long. Each target row's state.
    x_target_row : torch.Tensor
        (B,) long. Each state's selected row among the target rows, its "no target" row where
        none: the L3 selection.

    Returns
    -------
    PolicyOutput
        Each state's action and each level's statistics.
    """
    tokens_refined = core_output.tokens
    batch_size = tokens_refined.x_count.shape[0]

    # The target rows are the M Monsters, then one "no target" per state
    num_monsters = x_target_state.shape[0] - batch_size

    # Each level's log-prob, entropy and KL-Uniform
    x_log_prob_l1, x_entropy_l1, x_kl_uniform_l1 = _evaluate_idx(
        x_logit_action_type, x_action_type
    )
    x_log_prob_l2, x_entropy_l2, x_kl_uniform_l2 = _evaluate_row(
        x_logit_selection, core_output.x_token_state, batch_size, x_selection_row
    )
    x_log_prob_l3, x_entropy_l3, x_kl_uniform_l3 = _evaluate_row(
        x_logit_target, x_target_state, batch_size, x_target_row
    )

    # Calculate offsets 
    x_selection_off = x_selection_row - get_owner_first_row(tokens_refined.x_count)
    x_target_off = x_target_row - get_owner_first_row(core_output.x_total[:, TokenKind.MONSTERS])

    # Stack per-level columns: contiguous, faster than strided writes to a preallocated buffer
    x_action = torch.stack([x_action_type, x_selection_off, x_target_off], dim=1)
    x_log_prob = torch.stack([x_log_prob_l1, x_log_prob_l2, x_log_prob_l3], dim=1)
    x_entropy = torch.stack([x_entropy_l1, x_entropy_l2, x_entropy_l3], dim=1)
    x_kl_uniform = torch.stack([x_kl_uniform_l1, x_kl_uniform_l2, x_kl_uniform_l3], dim=1)

    # Whether each level applies: its selection is no placeholder, the Character or "no target"
    x_has_selection = tokens_refined.x_entity_kind[x_selection_row] != EntityKind.CHARACTER
    x_has_target = x_target_row < num_monsters
    x_level_applies = torch.stack(
        [torch.ones_like(x_has_selection), x_has_selection, x_has_target], dim=1
    )  # (B, 3)

    # A level a row doesn't reach: no selection (-1) and zero terms, whatever its placeholder drew
    return PolicyOutput(
        x_action=torch.where(x_level_applies, x_action, -1),
        x_log_prob=torch.where(x_level_applies, x_log_prob, 0.0),
        x_entropy=torch.where(x_level_applies, x_entropy, 0.0),
        x_kl_uniform=torch.where(x_level_applies, x_kl_uniform, 0.0),
    )


def _evaluate_idx(
    x_logit: torch.Tensor, x_idx_choice: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute each state's selected index's log-prob, with its entropy and KL-Uniform.

    Parameters
    ----------
    x_logit : torch.Tensor
        (B, C). A head's logits, C per state, -inf where illegal.
    x_idx_choice : torch.Tensor
        (B,) long. Each state's selected index.

    Returns
    -------
    x_log_prob : torch.Tensor
        (B,). The selected index's log-prob.
    x_entropy : torch.Tensor
        (B,). The entropy.
    x_kl_uniform : torch.Tensor
        (B,). KL-Uniform over the legal set.
    """
    x_legal = x_logit != float("-inf")

    # No argument checks: each one reads the logits back, making the CPU wait for the GPU
    dist = torch.distributions.Categorical(logits=x_logit, validate_args=False)

    # KL-Uniform over the legal set; the anti-collapse barrier
    x_num_legal = x_legal.sum(-1).float()
    x_logp = dist.logits.clamp(min=_KL_UNIFORM_LOG_P_FLOOR)
    x_kl_uniform = -x_logp.masked_fill(~x_legal, 0.0).sum(-1) / x_num_legal - x_num_legal.log()

    # The entropy over its maximum, log(legal count); min=2: a lone legal choice gives 0, not 0/0
    return (
        dist.log_prob(x_idx_choice),
        dist.entropy() / x_num_legal.clamp(min=2).log(),
        x_kl_uniform,
    )


def _evaluate_row(
    x_logit: torch.Tensor, x_row_state: torch.Tensor, batch_size: int, x_row: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute each state's selected row's log-prob, with its entropy and KL-Uniform.

    The same statistics as `_evaluate_idx`, over rows of logits grouped by state.

    Parameters
    ----------
    x_logit : torch.Tensor
        (R,). A head's logits, one per row, -inf where illegal.
    x_row_state : torch.Tensor
        (R,) long. Each row's state.
    batch_size : int
        The number of states, B.
    x_row : torch.Tensor
        (B,) long. Each state's selected row.

    Returns
    -------
    x_log_prob : torch.Tensor
        (B,). The selected row's log-prob.
    x_entropy : torch.Tensor
        (B,). The entropy.
    x_kl_uniform : torch.Tensor
        (B,). KL-Uniform over the legal set.
    """
    x_legal = x_logit != float("-inf")

    # Each row's log-prob within its state; the max keeps the exponentials finite, and takes no
    # gradient, as in logsumexp
    x_shifted = x_logit - _max_by_state(x_logit.detach(), x_row_state, batch_size)[x_row_state]
    x_sum_exp = _sum_by_state(x_shifted.exp(), x_row_state, batch_size)
    x_log_prob = x_shifted - x_sum_exp.log()[x_row_state]

    # The entropy over the legal set; a masked row's log-prob is -inf, so it sits out as 0
    x_log_prob_legal = x_log_prob.masked_fill(~x_legal, 0.0)
    x_entropy = -1 * _sum_by_state(x_log_prob_legal.exp() * x_log_prob_legal, x_row_state, batch_size)

    # KL-Uniform over the legal set, −log p clamped at the floor, as in `_evaluate_idx`
    x_num_legal = _sum_by_state(x_legal.to(x_logit.dtype), x_row_state, batch_size)
    x_log_prob_floor = x_log_prob.clamp(min=_KL_UNIFORM_LOG_P_FLOOR).masked_fill(~x_legal, 0.0)
    x_log_prob_sum = _sum_by_state(x_log_prob_floor, x_row_state, batch_size)
    x_kl_uniform = -1 * x_log_prob_sum / x_num_legal - x_num_legal.log()

    # The entropy over its maximum, log(legal count), as in `_evaluate_idx`
    return x_log_prob[x_row], x_entropy / x_num_legal.clamp(min=2).log(), x_kl_uniform


def _sum_by_state(
    x_value: torch.Tensor, x_row_state: torch.Tensor, batch_size: int
) -> torch.Tensor:
    """Sum each state's rows' `x_value`.

    Parameters
    ----------
    x_value : torch.Tensor
        (R,). Each row's value.
    x_row_state : torch.Tensor
        (R,) long. Each row's state.
    batch_size : int
        The number of states, B.

    Returns
    -------
    torch.Tensor
        (B,). Each state's sum of its rows' `x_value`.
    """
    return torch.zeros(batch_size, dtype=x_value.dtype, device=x_value.device).index_add_(
        0, x_row_state, x_value
    )


def _max_by_state(
    x_value: torch.Tensor, x_row_state: torch.Tensor, batch_size: int
) -> torch.Tensor:
    """Find each state's max of its rows' `x_value`.

    Parameters
    ----------
    x_value : torch.Tensor
        (R,). Each row's value.
    x_row_state : torch.Tensor
        (R,) long. Each row's state.
    batch_size : int
        The number of states, B.

    Returns
    -------
    torch.Tensor
        (B,). Each state's max of its rows' `x_value`.
    """
    return torch.full(
        (batch_size,), float("-inf"), dtype=x_value.dtype, device=x_value.device
    ).scatter_reduce_(0, x_row_state, x_value, reduce="amax")


def _argmax_by_state(
    x_value: torch.Tensor, x_row_state: torch.Tensor, batch_size: int
) -> torch.Tensor:
    """Find each state's row of its highest `x_value`, the first of a tie.

    Parameters
    ----------
    x_value : torch.Tensor
        (R,). Each row's value.
    x_row_state : torch.Tensor
        (R,) long. Each row's state.
    batch_size : int
        The number of states, B.

    Returns
    -------
    torch.Tensor
        (B,) long. Each state's row of its highest `x_value`, the first of a tie.
    """
    num_rows = x_value.shape[0]
    x_row = torch.arange(num_rows, device=x_value.device)
    x_is_max = x_value == _max_by_state(x_value, x_row_state, batch_size)[x_row_state]
    
    return torch.full(
        (batch_size,), num_rows, dtype=torch.long, device=x_value.device
    ).scatter_reduce_(0, x_row_state, x_row.masked_fill(~x_is_max, num_rows), reduce="amin")
