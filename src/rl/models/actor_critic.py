import math

import torch
import torch.nn as nn
from slai import ACTION_SPEC_REGISTRY

from src.rl.batches import get_entity_kind_rows
from src.rl.batches import get_owner_first_row
from src.rl.batches import get_row_owner
from src.rl.constants import ACTION_TYPES
from src.rl.constants import NUM_ACTION_TYPES
from src.rl.encoding import unpack_legal_bits
from src.rl.models.core import Core
from src.rl.models.heads import HeadActionType
from src.rl.models.heads import HeadTarget
from src.rl.models.heads import HeadSelection
from src.rl.models.heads import HeadValue
from src.rl.models.heads import PointerKeys
from src.rl.models.map_encoder import message_passing
from src.rl.reward import RewardStream
from src.rl.types import CollectionKind
from src.rl.types import CoreOutput
from src.rl.types import EntityKind
from src.rl.types import Live
from src.rl.types import MapCache
from src.rl.types import MapDedup
from src.rl.types import PolicyOutput
from src.rl.types import StateCollection
from src.rl.types import States
from src.rl.types import TokenDedup


# Barrier probability floor
_KL_UNIFORM_LOG_P_FLOOR = math.log(1e-4)


class ActorCritic(nn.Module):
    """The policy and the value function over a batch of game states.

    The policy has two levels: L1, the joint level, an action type with the token it selects, and
    L2, the target level, its Monster target. L1 is one categorical over each state's legal
    selections, (token, action type) pairs, and its legal action types that select nothing: a
    selection's logit is its action type's plus its token's, an action type that selects nothing
    scores its own. L2 runs after it, given its selections. Each head returns its logits.

    Each entry point runs the core once, then the heads its caller needs:
    - `action_select`: the Actor's step and the eval's. Selects each state's action, without
      values.
    - `action_evaluate`: the Learner's update. Scores recorded actions and computes the values.
    - `get_values`: the Learner's bootstrap. Computes the values alone.
    """

    _x_action_type_no_selection: torch.Tensor

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

        # Policy heads
        self._head_action_type = HeadActionType(dim_global, dim_ff_primary)
        self._head_selection = HeadSelection(dim_global, dim_op, dim_key)
        self._head_target = HeadTarget(dim_global, dim_entity, dim_key)

        # One key net for every token, shared by the L1 and L2 pointer selections
        self._pointer_keys = PointerKeys(dim_entity, dim_key)

        # Value head
        self._head_value = HeadValue(dim_global, dim_ff_value, len(RewardStream))

        # Buffer of the `ActionType`s that select nothing, each a joint row of every state
        self.register_buffer(
            "_x_action_type_no_selection",
            torch.tensor(
                [
                    idx_action_type
                    for idx_action_type, action_type in enumerate(ACTION_TYPES)
                    if ACTION_SPEC_REGISTRY[action_type].arity == (0, 0)
                ]
            ),
            persistent=False,
        )

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

        L1 selects one of its joint rows, which gives the action type and the selection; L2 then
        selects from its head's logits.

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
        batch_size = states.tokens.x_count.shape[0]
        num_selections = int(states.x_total_cpu[:, StateCollection.SELECTIONS].sum())
        num_monsters = int(states.x_total_cpu[:, CollectionKind.MONSTERS].sum())

        # Core pass
        core_output = self.core(states, map_cache=map_cache)
        tokens_refined = core_output.tokens

        # Calculate each token's pointer key
        x_token_key = self._pointer_keys(tokens_refined.features.x_value)  # (N, dim_key)

        # The joint rows and their logits, the legal selections counted on the host
        _, x_joint_token_row, x_joint_action_type, x_joint_state = _get_joint_rows(
            core_output, self._x_action_type_no_selection, num_selections
        )
        x_logit_joint = self._score_joint_rows(
            core_output,
            x_token_key,
            x_joint_token_row,
            x_joint_action_type,
            x_joint_state,
            num_selections,
        )

        # Select each state's joint row: its `ActionType` and selected token, its Character if none
        x_joint_row = _select_row(x_logit_joint, x_joint_state, batch_size, greedy, generator) # (B,)
        x_action_type = x_joint_action_type[x_joint_row]
        x_selection_row = x_joint_token_row[x_joint_row]

        # Each target row's state, the Monsters counted on the host
        x_target_state = _get_target_state(core_output, num_monsters)  # (M + B,)
        x_logit_target = self._head_target(
            core_output, x_token_key, x_target_state, x_action_type, x_selection_row
        )
        x_target_row = _select_row(x_logit_target, x_target_state, batch_size, greedy, generator)

        return self._evaluate_selections(
            core_output,
            x_logit_joint,
            x_joint_state,
            x_joint_row,
            x_action_type,
            x_selection_row,
            x_logit_target,
            x_target_state,
            x_target_row,
        )

    def action_evaluate(
        self,
        states: States[Live],
        x_recorded_action: torch.Tensor,
        map_dedup: MapDedup,
        token_dedup: TokenDedup | None,
    ) -> tuple[PolicyOutput, torch.Tensor]:
        """Score the recorded actions and compute the values.

        L1 scores its joint rows, each recorded action and its selection one of them; L2's head
        scores its choices given those selections.

        Parameters
        ----------
        states : States[Live]
            The game states.
        x_recorded_action : torch.Tensor
            (B, 3) long. Each state's recorded action: action type, selection offset within that
            state, Monster target rank; -1 where a selection or target didn't apply.
        map_dedup : MapDedup
            The Maps the Map GNN runs, planned on the host for an update (see `MapGNN.forward`).
        token_dedup : TokenDedup | None
            The unique tokens the projector runs, planned on the host for an update, or None to
            run every token (see `EntityProjector.forward`).

        Returns
        -------
        policy_output : PolicyOutput
            The recorded actions and each level's statistics.
        x_value : torch.Tensor
            (B, len(RewardStream)). The values.
        """
        num_selections = int(states.x_total_cpu[:, StateCollection.SELECTIONS].sum())
        num_monsters = int(states.x_total_cpu[:, CollectionKind.MONSTERS].sum())

        # Core pass
        core_output = self.core(states, map_dedup=map_dedup, token_dedup=token_dedup)
        tokens_refined = core_output.tokens

        # Value head
        x_value = self._head_value(core_output.x_global)

        # Calculate each token's pointer key
        x_token_key = self._pointer_keys(tokens_refined.features.x_value)  # (N, dim_key)

        # The joint rows, the legal selections counted on the host
        x_selection_flat, x_joint_token_row, x_joint_action_type, x_joint_state = _get_joint_rows(
            core_output, self._x_action_type_no_selection, num_selections
        )

        # Each target row's state, the Monsters counted on the host
        x_target_state = _get_target_state(core_output, num_monsters)  # (M + B,)

        # The recorded selections, each as a row of its level's logits
        x_action_type, x_selection_off, x_target = x_recorded_action.unbind(-1)
        x_selection_row = _row_of_selection_off(core_output, x_selection_off)
        x_joint_row = _row_of_joint(
            x_selection_flat,
            self._x_action_type_no_selection,
            x_action_type,
            x_selection_off,
            x_selection_row,
        )
        x_target_row = _row_of_target(core_output, num_monsters, x_target)

        # Each level's logits, given the recorded selections before it
        x_logit_joint = self._score_joint_rows(
            core_output,
            x_token_key,
            x_joint_token_row,
            x_joint_action_type,
            x_joint_state,
            num_selections,
        )
        x_logit_target = self._head_target(
            core_output, x_token_key, x_target_state, x_action_type, x_selection_row
        )

        return self._evaluate_selections(
            core_output,
            x_logit_joint,
            x_joint_state,
            x_joint_row,
            x_action_type,
            x_selection_row,
            x_logit_target,
            x_target_state,
            x_target_row,
        ), x_value

    def get_values(
        self, states: States[Live], map_dedup: MapDedup, token_dedup: TokenDedup | None
    ) -> torch.Tensor:
        """Compute the values.

        Parameters
        ----------
        states : States[Live]
            The game states.
        map_dedup : MapDedup
            The Maps the Map GNN runs, planned on the host for an update (see `MapGNN.forward`).
        token_dedup : TokenDedup | None
            The unique tokens the projector runs, planned on the host for an update, or None to
            run every token (see `EntityProjector.forward`).

        Returns
        -------
        torch.Tensor
            (B, len(RewardStream)). The values.
        """
        return self._head_value(
            self.core(states, map_dedup=map_dedup, token_dedup=token_dedup).x_global
        )

    def _score_joint_rows(
        self,
        core_output: CoreOutput,
        x_token_key: torch.Tensor,
        x_joint_token_row: torch.Tensor,
        x_joint_action_type: torch.Tensor,
        x_joint_state: torch.Tensor,
        num_selections: int,
    ) -> torch.Tensor:
        """Score the joint rows.

        A selection scores its `ActionType`'s logit plus its token's, from the selection head; an
        `ActionType` that selects nothing scores its own.

        Parameters
        ----------
        core_output : CoreOutput
            The core's output.
        x_token_key : torch.Tensor
            (N, dim_key). Each token's key, from PointerKeys.
        x_joint_token_row : torch.Tensor
            (P + B * K,) long. Each joint row's selected token, its state's Character where none.
        x_joint_action_type : torch.Tensor
            (P + B * K,) long. Each joint row's action type.
        x_joint_state : torch.Tensor
            (P + B * K,) long. Each joint row's state.
        num_selections : int
            The batch's number of legal selections, P, which are the first joint rows, as the host
            knows it.

        Returns
        -------
        torch.Tensor
            (P + B * K,). Each joint row's logit, -inf where its `ActionType` is illegal.
        """
        x_logit_action_type = self._head_action_type(core_output)  # (B, NUM_ACTION_TYPES)
        x_logit_joint = x_logit_action_type[x_joint_state, x_joint_action_type]  # (P + B * K,)

        # No legal selection: the compiled selection head is not handed zero rows
        if num_selections == 0:
            return x_logit_joint

        x_logit_selection = self._head_selection(
            core_output,
            x_token_key,
            x_joint_state[:num_selections],
            x_joint_token_row[:num_selections],
            x_joint_action_type[:num_selections],
        )  # (P,)

        return torch.cat(
            [x_logit_joint[:num_selections] + x_logit_selection, x_logit_joint[num_selections:]]
        )


def compile_hot_paths(model: ActorCritic) -> None:
    """Fuse the hot paths of the Learner's update and the Actor's rollout.

    They are each transformer block, the Map GNN's message passing, the key net, each policy head
    and the selections' statistics. The first two are autotuned, so the Learner's first update
    and the Actor's first rollout each compile for a minute or two.

    It compiles only on CUDA: a model on any other device stays eager.

    Parameters
    ----------
    model : ActorCritic
        The model, compiled in place.
    """
    if not next(model.parameters()).is_cuda:
        return

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


def _get_joint_rows(
    core_output: CoreOutput, x_action_type_no_selection: torch.Tensor, num_selections: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find each joint row's selected token, action type and state.

    The joint rows are the legal selections, in flat (token, action type) order, then each state's
    K action types that select nothing, state after state, with its Character as their selection.

    Parameters
    ----------
    core_output : CoreOutput
        The core's output.
    x_action_type_no_selection : torch.Tensor
        (K,) long. The action types that select nothing, ascending.
    num_selections : int
        The batch's number of legal selections, P, as the host knows it: no size is read back from
        the device.

    Returns
    -------
    x_selection_flat : torch.Tensor
        (P,) long. Each legal selection's flat index among the selection masks' bits, token row *
        NUM_ACTION_TYPES + action type, ascending.
    x_joint_token_row : torch.Tensor
        (P + B * K,) long. Each joint row's selected token, its state's Character where none.
    x_joint_action_type : torch.Tensor
        (P + B * K,) long. Each joint row's action type.
    x_joint_state : torch.Tensor
        (P + B * K,) long. Each joint row's state.
    """
    tokens = core_output.tokens
    batch_size = tokens.x_count.shape[0]
    num_action_types_no_selection = x_action_type_no_selection.shape[0]

    # Each token's selection bit of each action type, then the set ones as flat indexes: (P,)
    x_selection_bit = unpack_legal_bits(tokens.x_legal_selection)
    x_selection_flat = torch.nonzero_static(
        x_selection_bit.flatten(), size=num_selections
    ).squeeze(1)

    # Each state's Character, the selection of its action types that select nothing
    x_character_row = get_entity_kind_rows(
        tokens.x_entity_kind, EntityKind.CHARACTER, batch_size
    )  # (B,)
    x_joint_token_row = torch.cat(
        [
            x_selection_flat // NUM_ACTION_TYPES,
            x_character_row.repeat_interleave(num_action_types_no_selection),
        ]
    )
    x_joint_action_type = torch.cat(
        [x_selection_flat % NUM_ACTION_TYPES, x_action_type_no_selection.repeat(batch_size)]
    )
    x_joint_state = core_output.x_token_state[x_joint_token_row]

    return x_selection_flat, x_joint_token_row, x_joint_action_type, x_joint_state


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
    
    x_count_monster = core_output.x_total[:, CollectionKind.MONSTERS]
    x_monster_state = get_row_owner(x_count_monster, num_monsters)  # (M,)
    x_idx_state = torch.arange(batch_size, device=x_monster_state.device)

    return torch.cat([x_monster_state, x_idx_state])


def _row_of_joint(
    x_selection_flat: torch.Tensor,
    x_action_type_no_selection: torch.Tensor,
    x_action_type: torch.Tensor,
    x_selection_off: torch.Tensor,
    x_selection_row: torch.Tensor,
) -> torch.Tensor:
    """Find each state's action type `x_action_type` and its selection among the joint rows.

    Parameters
    ----------
    x_selection_flat : torch.Tensor
        (P,) long. Each legal selection's flat index among the selection masks' bits, ascending:
        the first P joint rows.
    x_action_type_no_selection : torch.Tensor
        (K,) long. The action types that select nothing, ascending: each state's K joint rows
        after the selections.
    x_action_type : torch.Tensor
        (B,) long. Each state's action type.
    x_selection_off : torch.Tensor
        (B,) long. Each state's selection offset in its state, -1 where none.
    x_selection_row : torch.Tensor
        (B,) long. Each state's selected token's row among the batch's tokens.

    Returns
    -------
    torch.Tensor
        (B,) long. Each state's joint row: its selection's, or its action type's among its
        state's rows after the selections where the offset is -1.
    """
    batch_size = x_action_type.shape[0]
    num_selections = x_selection_flat.shape[0]
    num_action_types_no_selection = x_action_type_no_selection.shape[0]

    x_idx_state = torch.arange(batch_size, device=x_action_type.device)

    # A selection's row is its flat index's rank among the selections, which ascend
    x_joint_row_selection = torch.searchsorted(
        x_selection_flat, x_selection_row * NUM_ACTION_TYPES + x_action_type
    )

    # An action type that selects nothing: its state's rows after the selections, then its rank;
    # `searchsorted` warns on a strided column, as a recorded one is
    x_joint_row_selection_none = (
        num_selections
        + x_idx_state * num_action_types_no_selection
        + torch.searchsorted(x_action_type_no_selection, x_action_type.contiguous())
    )

    return torch.where(x_selection_off >= 0, x_joint_row_selection, x_joint_row_selection_none)


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
    x_count_monster = core_output.x_total[:, CollectionKind.MONSTERS]
    x_state_first_monster = get_owner_first_row(x_count_monster)  # (B,)

    return torch.where(x_target >= 0, x_state_first_monster + x_target, num_monsters + x_idx_state)


def _select_row(
    x_logit: torch.Tensor,
    x_row_state: torch.Tensor,
    batch_size: int,
    greedy: bool,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Select each state's row among its rows of logits.

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
    x_logit_joint: torch.Tensor,
    x_joint_state: torch.Tensor,
    x_joint_row: torch.Tensor,
    x_action_type: torch.Tensor,
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
    x_logit_joint : torch.Tensor
        (P + B * K,). The L1 logits, one per joint row.
    x_joint_state : torch.Tensor
        (P + B * K,) long. Each joint row's state.
    x_joint_row : torch.Tensor
        (B,) long. Each state's selected joint row: the L1 selection.
    x_action_type : torch.Tensor
        (B,) long. Each state's action type, its joint row's.
    x_selection_row : torch.Tensor
        (B,) long. Each state's selected token's row among the tokens, its Character's where
        none: its joint row's.
    x_logit_target : torch.Tensor
        (M + B,). The L2 logits, one per target row.
    x_target_state : torch.Tensor
        (M + B,) long. Each target row's state.
    x_target_row : torch.Tensor
        (B,) long. Each state's selected row among the target rows, its "no target" row where
        none: the L2 selection.

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
    x_log_prob_l1, x_entropy_l1, x_kl_uniform_l1 = _evaluate_row(
        x_logit_joint, x_joint_state, batch_size, x_joint_row
    )
    x_log_prob_l2, x_entropy_l2, x_kl_uniform_l2 = _evaluate_row(
        x_logit_target, x_target_state, batch_size, x_target_row
    )

    # Calculate offsets 
    x_selection_off = x_selection_row - get_owner_first_row(tokens_refined.x_count)
    x_target_off = x_target_row - get_owner_first_row(
        core_output.x_total[:, CollectionKind.MONSTERS]
    )

    # Stack the columns: contiguous, faster than strided writes to a preallocated buffer
    x_action = torch.stack([x_action_type, x_selection_off, x_target_off], dim=1)
    x_log_prob = torch.stack([x_log_prob_l1, x_log_prob_l2], dim=1)
    x_entropy = torch.stack([x_entropy_l1, x_entropy_l2], dim=1)
    x_kl_uniform = torch.stack([x_kl_uniform_l1, x_kl_uniform_l2], dim=1)

    # Whether each coordinate and level applies: its selection is no placeholder, the Character
    # or "no target"
    x_has_selection = tokens_refined.x_entity_kind[x_selection_row] != EntityKind.CHARACTER
    x_has_target = x_target_row < num_monsters
    x_level_applies = torch.stack([torch.ones_like(x_has_target), x_has_target], dim=1)  # (B, 2)
    x_coordinate_applies = torch.stack(
        [torch.ones_like(x_has_selection), x_has_selection, x_has_target], dim=1
    )  # (B, 3)

    # A coordinate or level a row doesn't reach: no selection (-1) and zero terms, whatever its
    # placeholder drew
    return PolicyOutput(
        x_action=torch.where(x_coordinate_applies, x_action, -1),
        x_log_prob=torch.where(x_level_applies, x_log_prob, 0.0),
        x_entropy=torch.where(x_level_applies, x_entropy, 0.0),
        x_kl_uniform=torch.where(x_level_applies, x_kl_uniform, 0.0),
    )


def _evaluate_row(
    x_logit: torch.Tensor, x_row_state: torch.Tensor, batch_size: int, x_row: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute each state's selected row's log-prob, with its entropy and KL-Uniform.

    The same statistics, over rows of logits grouped by state.

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

    # Each row's log-prob within its state; the max keeps exponentials finite, and takes no
    # gradient, as in logsumexp
    x_shifted = x_logit - _max_by_state(x_logit.detach(), x_row_state, batch_size)[x_row_state]
    x_sum_exp = _sum_by_state(x_shifted.exp(), x_row_state, batch_size)
    x_log_prob = x_shifted - x_sum_exp.log()[x_row_state]

    # The entropy over the legal set; a masked row's log-prob is -inf, so it sits out as 0
    x_log_prob_legal = x_log_prob.masked_fill(~x_legal, 0.0)
    x_entropy = -1 * _sum_by_state(x_log_prob_legal.exp() * x_log_prob_legal, x_row_state, batch_size)

    # KL-Uniform over the legal set, −log p clamped at the floor; the anti-collapse barrier
    x_num_legal = _sum_by_state(x_legal.to(x_logit.dtype), x_row_state, batch_size)
    x_log_prob_floor = x_log_prob.clamp(min=_KL_UNIFORM_LOG_P_FLOOR).masked_fill(~x_legal, 0.0)
    x_log_prob_sum = _sum_by_state(x_log_prob_floor, x_row_state, batch_size)
    x_kl_uniform = -1 * x_log_prob_sum / x_num_legal - x_num_legal.log()

    # The entropy over its maximum, log(legal count); min=2: a lone legal choice gives 0, not 0/0
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
