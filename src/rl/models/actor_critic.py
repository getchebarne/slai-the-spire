import math
from collections.abc import Callable

import torch
import torch.nn as nn
from slai import ACTION_SPEC_REGISTRY

from src.rl.encoding.focus import ENCODING_DIM_FOCUS
from src.rl.encoding.state import COL_START_CHARACTER
from src.rl.encoding.state import COL_START_EVENT_OPTION
from src.rl.encoding.state import COL_START_HEALTH_BLOCK
from src.rl.encoding.state import COL_START_MONSTER
from src.rl.encoding.state import COL_START_POTION
from src.rl.encoding.state import COL_START_PRICE
from src.rl.encoding.state import COL_START_RELIC
from src.rl.models.core import Core
from src.rl.models.heads import HeadActionType
from src.rl.models.heads import HeadPointerSelect
from src.rl.models.heads import HeadValue
from src.rl.models.heads import PointerKeys
from src.rl.models.map_encoder import message_passing
from src.rl.reward import RewardStream
from src.rl.types import ACTION_TYPE_BY_INT
from src.rl.types import NUM_ACTION_TYPES
from src.rl.types import TokenKind
from src.rl.types import X_Action
from src.rl.types import X_CoreOutput
from src.rl.types import X_GameState
from src.rl.types import X_Window


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
    """Each row's selection, its log-prob, the entropy and KL(uniform‖π), all (B,), from a head's
    (B, N) logits, which the head already masked; `x_mask` is the KL's legal set. Samples draw
    from `generator`, or the global one when it is None"""
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


def _sum_by_state(x: torch.Tensor, x_state: torch.Tensor, num_states: int) -> torch.Tensor:
    """Each state's sum of its rows' `x`, (S,)"""
    return x.new_zeros(num_states).index_add_(0, x_state, x)


def _max_by_state(x: torch.Tensor, x_state: torch.Tensor, num_states: int) -> torch.Tensor:
    """Each state's max of its rows' `x`, (S,)"""
    return x.new_full((num_states,), float("-inf")).scatter_reduce_(0, x_state, x, reduce="amax")


def _argmax_by_state(x: torch.Tensor, x_state: torch.Tensor, num_states: int) -> torch.Tensor:
    """Each state's row of its highest `x`, the first of a tie, (S,)"""
    num_rows = x.shape[0]
    x_row = torch.arange(num_rows, device=x.device)
    x_is_max = x == _max_by_state(x, x_state, num_states)[x_state]
    return x_row.new_full((num_states,), num_rows).scatter_reduce_(
        0, x_state, x_row.masked_fill(~x_is_max, num_rows), reduce="amin"
    )


def _categorical_rows(
    x_logits: torch.Tensor,
    x_mask: torch.Tensor,
    x_state: torch.Tensor,
    num_states: int,
    x_recorded_row: torch.Tensor | None,
    greedy: bool,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`_categorical_step` over packed rows: each state's selected row, its log-prob, the entropy
    and KL(uniform‖π), all (S,), from its rows' logits (R,), which the head already masked;
    `x_mask` is the KL's legal set and `x_state` each row's state. Samples are Gumbel-max: one
    draw per row from `generator`, or the global one when it is None, then each state's argmax"""
    # Each row's log-prob within its state; the max keeps the exponentials finite, and takes no
    # gradient, as in logsumexp
    x_shifted = x_logits - _max_by_state(x_logits.detach(), x_state, num_states)[x_state]
    x_log_prob = x_shifted - _sum_by_state(x_shifted.exp(), x_state, num_states).log()[x_state]

    # Determine which row to select
    if x_recorded_row is not None:
        # PPO recompute path
        x_row = x_recorded_row
    elif greedy:
        # Highest logit
        x_row = _argmax_by_state(x_logits, x_state, num_states)
    else:
        # Sample: Gumbel noise per row, masked after it is added, then each state's highest
        x_gumbel = -torch.empty_like(x_logits).exponential_(generator=generator).log()
        x_score = (x_logits + x_gumbel).masked_fill(~x_mask, float("-inf"))
        x_row = _argmax_by_state(x_score, x_state, num_states)

    # The entropy over the legal set; a masked row's log-prob is -inf, so it sits out as 0
    x_log_prob_legal = x_log_prob.masked_fill(~x_mask, 0.0)
    x_entropy = -_sum_by_state(x_log_prob_legal.exp() * x_log_prob_legal, x_state, num_states)

    # KL(uniform‖π) over the legal set, −log p clamped at the floor, as in `_categorical_step`;
    # min=1: empty/singleton states → KL 0
    x_n_legal = _sum_by_state(x_mask.to(x_logits.dtype), x_state, num_states).clamp(min=1)
    x_log_prob_floor = x_log_prob.clamp(min=_KL_UNIFORM_LOG_P_FLOOR).masked_fill(~x_mask, 0.0)
    x_log_prob_sum = _sum_by_state(x_log_prob_floor, x_state, num_states)
    x_kl_uniform = -x_log_prob_sum / x_n_legal - x_n_legal.log()
    return x_row, x_log_prob[x_row], x_entropy, x_kl_uniform


def _row(x_window: X_Window, x_token_kind: torch.Tensor, x_rank: torch.Tensor) -> torch.Tensor:
    """The row of the rank-th token of `token_kind`, per state; where the token kind is -1, the
    placeholder's: its state's character row."""
    x_first_row = x_window.x_token_kind_row.gather(1, x_token_kind.clamp(min=0).unsqueeze(1))
    return torch.where(
        x_token_kind >= 0,
        x_first_row.squeeze(1) + x_rank,
        x_window.x_token_kind_row[:, TokenKind.CHARACTER],
    )


def _token_kind_rank(x_window: X_Window, x_row: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A row's token kind and its rank within that token kind, per state."""
    x_token_kind = x_window.x_token_kind[x_row]
    x_first_row = x_window.x_token_kind_row.gather(1, x_token_kind.unsqueeze(1)).squeeze(1)
    return x_token_kind, x_row - x_first_row


class ActorCritic(nn.Module):
    # The buffers __init__ registers, typed for mypy
    _selects_token: torch.Tensor
    _is_targeted: torch.Tensor

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

        # L2/L3 pointer selection: one key net per entity class + a shared query net per level
        self._action_type_embedding = nn.Embedding(NUM_ACTION_TYPES, dim_op)

        # One key net per entity class; every TokenKind of that class shares it, so an entity maps
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
        arities = [ACTION_SPEC_REGISTRY[at].arity for at in ACTION_TYPE_BY_INT]
        self.register_buffer(
            "_selects_token",
            torch.tensor([arity != (0, 0) for arity in arities]),
            persistent=False,
        )
        self.register_buffer(
            "_is_targeted", torch.tensor([arity[1] == 2 for arity in arities]), persistent=False
        )

        # The policy heads' computation as a plain function, which compile_hot_paths swaps for
        # its compiled version on the learner's model
        self.policy_heads_recorded = _policy_heads

        self._init_weights()

    def _init_weights(self) -> None:
        """PPO init: orthogonal (√2), near-zero logit layers (policy ≈ uniform), unit-gain value"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:  # Some projector layers are bias-free
                    nn.init.zeros_(m.bias)

        # The shared first layer per column range, each family's then the health/block and price
        # ones, as its own layer: orthogonal over all of them would shrink each column's scale
        with torch.no_grad():
            for x_weight_range in torch.tensor_split(
                self.core._entity_projector._pre_token.weight,
                (
                    COL_START_RELIC,
                    COL_START_POTION,
                    COL_START_MONSTER,
                    COL_START_EVENT_OPTION,
                    COL_START_CHARACTER,
                    COL_START_HEALTH_BLOCK,
                    COL_START_PRICE,
                ),
                dim=1,
            ):
                nn.init.orthogonal_(x_weight_range, gain=math.sqrt(2))
        nn.init.orthogonal_(self._head_action_type._scorer[-1].weight, gain=0.01)
        nn.init.orthogonal_(self._head_level_2._query_net[-1].weight, gain=0.01)
        nn.init.orthogonal_(self._head_level_3._query_net[-1].weight, gain=0.01)
        nn.init.orthogonal_(self._head_value._network[-1].weight, gain=1.0)

    def _select_actions(
        self,
        heads: Callable[..., tuple[torch.Tensor, ...]],
        x_gs: X_GameState,
        x_core_out: X_CoreOutput,
        greedy: bool,
        generator: torch.Generator | None = None,
        x_recorded_actions: torch.Tensor | None = None,
    ) -> X_Action:
        x_action_type, x_token_kind, x_rank, x_target, x_log_prob, x_entropy, x_kl_uniform = heads(
            self,
            x_focus=x_gs.focus,
            x_global=x_core_out.global_,
            x_tokens=x_core_out.x_tokens,
            x_window=x_core_out.window,
            x_legal_action_type=x_gs.x_legal_action_type,
            greedy=greedy,
            generator=generator,
            x_recorded_actions=x_recorded_actions,
        )
        return X_Action(
            action_type=x_action_type,
            token_kind=x_token_kind,
            rank=x_rank,
            target=x_target,
            log_prob=x_log_prob,
            entropy=x_entropy,
            kl_uniform=x_kl_uniform,
        )

    def forward(self, x_gs: X_GameState) -> torch.Tensor:
        """The (B, len(RewardStream)) values"""
        return self._head_value(self.core(x_gs).global_value)

    def act(
        self,
        x_gs: X_GameState,
        greedy: bool = False,
        generator: torch.Generator | None = None,
    ) -> X_Action:
        """The X_Action, each level's argmax under `greedy` and sampled from `generator`
        otherwise; without values to compute, the core runs the draw-order layer only on the states
        the policy reads"""
        x_core_out = self.core(x_gs, with_value=False)

        # Sampling actions runs eager
        return self._select_actions(_policy_heads, x_gs, x_core_out, greedy, generator)

    def evaluate_actions(
        self, x_gs: X_GameState, x_recorded_actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The recorded actions' log-prob summed over levels (B,), each level's entropy and
        KL(uniform‖π) (B, 3), and the (B, len(RewardStream)) values"""
        x_core_out = self.core(x_gs)
        x_values = self._head_value(x_core_out.global_value)

        # Recorded actions run the attribute, compiled on the learner
        x_action = self._select_actions(
            self.policy_heads_recorded,
            x_gs,
            x_core_out,
            greedy=True,
            x_recorded_actions=x_recorded_actions,
        )
        return x_action.log_prob.sum(-1), x_action.entropy, x_action.kl_uniform, x_values


def _policy_heads(
    model: ActorCritic,
    x_focus: torch.Tensor,
    x_global: torch.Tensor,
    x_tokens: torch.Tensor,
    x_window: X_Window,
    x_legal_action_type: torch.Tensor,
    greedy: bool,
    generator: torch.Generator | None,
    x_recorded_actions: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    """`model`'s L1-L3 selections, or the recorded ones scored, over the core's global vector and
    refined rows of the packed window, given the L1 mask. L2 and L3 select per state over its rows,
    grouped by family. Returns X_Action's fields in order"""
    # The recorded actions' columns, or None when selecting
    x_recorded_action_type, x_recorded_token_kind, x_recorded_rank, x_recorded_target = (
        x_recorded_actions.unbind(-1) if x_recorded_actions is not None else (None,) * 4
    )

    # Select the `ActionType` for every row — one masked categorical, focus-gated.
    x_logits = model._head_action_type(x_global, x_focus, x_legal_action_type)
    x_action_type, x_log_prob_l1, x_entropy_l1, x_kl_uniform_l1 = _categorical_step(
        x_logits, x_legal_action_type, x_recorded_action_type, greedy, generator
    )

    # Compute the `Action`'s selections from the chosen `ActionType`
    batch_size = x_action_type.shape[0]
    x_states = torch.arange(batch_size, device=x_action_type.device)

    # The rows grouped by family, family after family and each family's state after state: what
    # the key nets split and the scoring runs on. Selections map back to the window through it
    x_order = torch.argsort(x_window.x_family, stable=True)  # (N,)
    x_state = x_window.x_state[x_order]
    (
        num_cards,
        num_relics,
        num_potions,
        num_monsters,
        num_event_options,
        num_character,
        _,
    ) = x_window.num_tokens_family

    # The split hands each class its rows
    x_cards, x_relics, x_potions, x_monster_tokens, x_event_options, _, x_rooms = x_tokens[
        x_order
    ].split(x_window.num_tokens_family)

    # L2 keys, once per token: its own class's net, the one every action type scores
    # it with. Monsters and the character are never L2 selections: zero keys
    x_keys = torch.cat(
        [
            model._pointer_key_card(x_cards),
            model._pointer_key_relic(x_relics),
            model._pointer_key_potion(x_potions),
            x_tokens.new_zeros(num_monsters, model._dim_key),
            model._pointer_key_event(x_event_options),
            x_tokens.new_zeros(num_character, model._dim_key),
            model._pointer_key_room(x_rooms),
        ]
    )

    # L2: score every row against its state's query, conditioned on the `ActionType` embedding;
    # the mask keeps exactly the type's legal tokens
    x_selects_token = model._selects_token[x_action_type]
    x_legal = x_window.x_legal.gather(1, x_action_type[x_window.x_state].unsqueeze(1))
    x_mask_rows = x_legal.squeeze(1)[x_order]

    # Every state at once: a type that selects no token (TurnEnd...) gets its state's character
    # row as a placeholder, so its categorical stays finite. One per state, past the earlier
    # families' rows
    num_rows_before_character = (
        num_cards + num_relics + num_potions + num_monsters + num_event_options
    )
    x_mask_rows[num_rows_before_character + x_states] |= ~x_selects_token

    # Recorded selections are (token kind, rank); the pointer scores rows
    # A state without a selection records token kind and rank -1: _row gives the placeholder
    x_recorded_row = None
    if x_recorded_token_kind is not None:
        # Each window row's place among the grouped rows
        x_place = torch.empty_like(x_order)
        x_place[x_order] = torch.arange(x_order.shape[0], device=x_order.device)
        x_recorded_row = x_place[_row(x_window, x_recorded_token_kind, x_recorded_rank)]

    # Calculate the rows' logits & select from each state's
    x_logits = model._head_level_2(
        x_keys, x_global, model._action_type_embedding(x_action_type), x_state, x_mask_rows
    )
    x_row_selected, x_log_prob_l2, x_entropy_l2, x_kl_uniform_l2 = _categorical_rows(
        x_logits, x_mask_rows, x_state, batch_size, x_recorded_row, greedy, generator
    )

    # The selected rows back in the window
    x_token_selected = x_order[x_row_selected]
    x_token_kind, x_rank = _token_kind_rank(x_window, x_token_selected)

    # L3: monster target, only for actions that take one (CardPlay / PotionUse)
    # Keep only rows whose committed L2 selection takes a target: at recompute
    # the recorded selection says so (-1 = none), at sample time the mask does
    x_has_target = (
        x_recorded_target >= 0
        if x_recorded_target is not None
        else model._is_targeted[x_action_type] & x_window.x_needs_target[x_token_selected]
    )

    # The choices: each monster row, scored against its state's selected card/potion token and
    # legal where its state takes a target; then each state's "no target" choice, a zero key,
    # legal where it takes none
    x_monster_state = x_state.narrow(0, num_cards + num_relics + num_potions, num_monsters)
    x_choice_state = torch.cat([x_monster_state, x_states])
    x_mask_l3 = torch.cat([x_has_target[x_monster_state], ~x_has_target])

    # A cat, not F.pad: compiled, F.pad's backward fails on an empty monster block
    x_keys = torch.cat(
        [
            model._pointer_key_monster(x_monster_tokens),
            x_monster_tokens.new_zeros(batch_size, model._dim_key),
        ]
    )

    # A target is a monster's rank: its choice is its state's first monster choice plus the rank
    x_count_monster = x_window.x_token_kind_count[:, TokenKind.MONSTERS]
    x_first_monster = x_count_monster.cumsum(0) - x_count_monster
    x_recorded_choice = None
    if x_recorded_target is not None:
        x_recorded_choice = torch.where(
            x_has_target, x_first_monster + x_recorded_target, num_monsters + x_states
        )

    # Calculate the choices' logits & select from each state's
    x_logits = model._head_level_3(
        x_keys, x_global, x_tokens[x_token_selected], x_choice_state, x_mask_l3
    )
    x_choice, x_log_prob_l3, x_entropy_l3, x_kl_uniform_l3 = _categorical_rows(
        x_logits, x_mask_l3, x_choice_state, batch_size, x_recorded_choice, greedy, generator
    )
    x_target = x_choice - x_first_monster

    # A level a row doesn't reach: no selection (-1) and zero terms, whatever its placeholder drew
    x_token_kind = torch.where(x_selects_token, x_token_kind, -1)
    x_rank = torch.where(x_selects_token, x_rank, -1)
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
        x_token_kind,
        x_rank,
        x_target,
        torch.stack([x_log_prob_l1, x_log_prob_l2, x_log_prob_l3], dim=1),
        torch.stack([x_entropy_l1, x_entropy_l2, x_entropy_l3], dim=1),
        torch.stack([x_kl_uniform_l1, x_kl_uniform_l2, x_kl_uniform_l3], dim=1),
    )


def compile_hot_paths(model: ActorCritic) -> None:
    """Fuse the update's hot paths: each transformer block, the draw-order layer's block, the map
    GNN's message passing, and the policy heads on recorded actions. All but the heads are
    autotuned, so the first update compiles for a minute or two"""
    mode = "max-autotune-no-cudagraphs"
    for block in model.core._entity_transformer._entity_transformer_blocks:
        block.compile(dynamic=True, mode=mode)
    model.core._draw_order_layer._draw_order_block.compile(dynamic=True, mode=mode)
    model.core._map_gnn.message_passing = torch.compile(message_passing, dynamic=True, mode=mode)

    # Not autotuned: an empty block's key-net matmul fails to lower under autotuning
    model.policy_heads_recorded = torch.compile(_policy_heads, dynamic=True)
