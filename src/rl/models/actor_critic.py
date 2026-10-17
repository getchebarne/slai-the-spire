import math
from collections.abc import Callable

import torch
import torch.nn as nn
from slai import ACTION_SPEC_REGISTRY

from src.rl.encoding.focus import ENCODING_DIM_FOCUS
from src.rl.models.core import Core
from src.rl.models.heads import HeadActionType
from src.rl.models.heads import HeadPointerSelect
from src.rl.models.heads import HeadValue
from src.rl.models.heads import PointerKeys
from src.rl.models.map_encoder import message_passing
from src.rl.reward import RewardStream
from src.rl.types import ACTION_TYPE_BY_INT
from src.rl.types import NUM_ACTION_TYPES
from src.rl.types import X_Action
from src.rl.types import X_CoreOutput
from src.rl.types import X_GameState
from src.rl.types import X_Mask


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


def _position(
    x_slice_kind: torch.Tensor, x_kind: torch.Tensor, x_rank: torch.Tensor
) -> torch.Tensor:
    """The window position of the rank-th token of `kind`, per row."""
    x_match = x_slice_kind == x_kind.unsqueeze(1)
    x_hit = x_match & (x_match.cumsum(1) == x_rank.unsqueeze(1) + 1)
    return x_hit.long().argmax(1)


def _kind_rank(
    x_slice_kind: torch.Tensor, x_pos: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A window position's kind and its rank within that kind, per row."""
    x_kind = x_slice_kind.gather(1, x_pos.unsqueeze(1)).squeeze(1)
    x_before = torch.arange(x_slice_kind.shape[1], device=x_slice_kind.device) < x_pos.unsqueeze(1)
    return x_kind, ((x_slice_kind == x_kind.unsqueeze(1)) & x_before).sum(1)


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

        # One key net per entity class; every SliceKind of that class shares it, so an entity maps
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
        self.policy_heads_recorded = policy_heads

        self._init_weights()

    def _init_weights(self) -> None:
        """PPO init: orthogonal (√2), near-zero logit layers (policy ≈ uniform), unit-gain value"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:  # Some projector layers are bias-free
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.MultiheadAttention):
                # in_proj is a raw Parameter, not a Linear — the isinstance walk
                # above misses it, which would leave the Q/K/V projections at xavier
                nn.init.orthogonal_(m.in_proj_weight, gain=math.sqrt(2))
                nn.init.zeros_(m.in_proj_bias)
        nn.init.orthogonal_(self._head_action_type._scorer[-1].weight, gain=0.01)
        nn.init.orthogonal_(self._head_level_2._query_net[-1].weight, gain=0.01)
        nn.init.orthogonal_(self._head_level_3._query_net[-1].weight, gain=0.01)
        nn.init.orthogonal_(self._head_value._network[-1].weight, gain=1.0)

    def _select_actions(
        self,
        heads: Callable[..., tuple[torch.Tensor, ...]],
        x_focus: torch.Tensor,
        x_core_out: X_CoreOutput,
        x_mask: X_Mask,
        greedy: bool,
        generator: torch.Generator | None = None,
        x_recorded_actions: torch.Tensor | None = None,
    ) -> X_Action:
        # The masks over the window: the blocks cat in field order, exactly as the tokens
        blocks = (
            x_mask.cards,
            x_mask.relics,
            x_mask.potions,
            x_mask.monsters,
            x_mask.event_options,
            x_mask.character,
            x_mask.rooms,
        )
        x_action_type, x_kind, x_rank, x_target, x_log_prob, x_entropy, x_kl_uniform = heads(
            self,
            x_focus=x_focus,
            x_global=x_core_out.global_,
            x_tokens=x_core_out.entities.x,
            x_slice_kind=x_core_out.entities.x_slice_kind,
            x_mask_action_type=x_mask.mask_action_type,
            x_legal_blocks=tuple(block.legal for block in blocks),
            x_needs_target_blocks=tuple(block.needs_target for block in blocks),
            x_monster_is_target=x_mask.monster_is_target,
            greedy=greedy,
            generator=generator,
            x_recorded_actions=x_recorded_actions,
        )
        return X_Action(
            action_type=x_action_type,
            kind=x_kind,
            rank=x_rank,
            target=x_target,
            log_prob=x_log_prob,
            entropy=x_entropy,
            kl_uniform=x_kl_uniform,
            batch_size=[x_action_type.shape[0]],
        )

    def forward(self, x_game_state: X_GameState) -> torch.Tensor:
        """The (B, len(RewardStream)) values"""
        return self._head_value(self.core(x_game_state).global_value)

    def act(
        self,
        x_game_state: X_GameState,
        x_mask: X_Mask,
        greedy: bool = False,
        generator: torch.Generator | None = None,
    ) -> X_Action:
        """The X_Action, each level's argmax under `greedy` and sampled from `generator`
        otherwise; without values to compute, the core runs the draw-order layer only on the rows
        the policy reads"""
        x_core_out = self.core(x_game_state, with_value=False)

        # Sampling actions runs eager
        return self._select_actions(
            policy_heads, x_game_state.focus, x_core_out, x_mask, greedy, generator
        )

    def evaluate_actions(
        self, x_game_state: X_GameState, x_mask: X_Mask, x_recorded_actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The recorded actions' log-prob summed over levels (B,), each level's entropy and
        KL(uniform‖π) (B, 3), and the (B, len(RewardStream)) values"""
        x_core_out = self.core(x_game_state)
        x_values = self._head_value(x_core_out.global_value)

        # Recorded actions run the attribute, compiled on the learner
        x_action = self._select_actions(
            self.policy_heads_recorded,
            x_game_state.focus,
            x_core_out,
            x_mask,
            greedy=True,
            x_recorded_actions=x_recorded_actions,
        )
        return x_action.log_prob.sum(-1), x_action.entropy, x_action.kl_uniform, x_values


def policy_heads(
    model: ActorCritic,
    x_focus: torch.Tensor,
    x_global: torch.Tensor,
    x_tokens: torch.Tensor,
    x_slice_kind: torch.Tensor,
    x_mask_action_type: torch.Tensor,
    x_legal_blocks: tuple[torch.Tensor, ...],
    x_needs_target_blocks: tuple[torch.Tensor, ...],
    x_monster_is_target: torch.Tensor,
    greedy: bool,
    generator: torch.Generator | None,
    x_recorded_actions: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    """`model`'s L1-L3 selections, or the recorded ones scored, over the core's global vector and
    refined window (tokens and kinds), given the L1 mask, each window block's `legal` and
    `needs_target` masks in window order, and the monster block's `monster_is_target`. Returns
    X_Action's fields in order"""
    # The recorded actions' columns, or None when selecting
    x_recorded_action_type, x_recorded_kind, x_recorded_rank, x_recorded_target = (
        x_recorded_actions.unbind(-1) if x_recorded_actions is not None else (None,) * 4
    )

    # Select the `ActionType` for every row — one masked categorical, focus-gated.
    x_logits = model._head_action_type(x_global, x_focus, x_mask_action_type)
    x_action_type, x_log_prob_l1, x_entropy_l1, x_kl_uniform_l1 = _categorical_step(
        x_logits, x_mask_action_type, x_recorded_action_type, greedy, generator
    )

    # Compute the `Action`'s selections from the chosen `ActionType`
    batch_size = x_action_type.shape[0]
    x_rows = torch.arange(batch_size, device=x_action_type.device)

    # The masks over the whole window
    x_legal = torch.cat(x_legal_blocks, dim=-1)
    x_needs_target = torch.cat(x_needs_target_blocks, dim=-1)

    # The window's blocks, as wide as their masks: Python ints, so cutting the window reads
    # nothing back from the GPU
    widths = [x_block.shape[1] for x_block in x_needs_target_blocks]

    # Contiguous copies: a view would keep the whole window alive for the backward
    (
        x_cards,
        x_relics,
        x_potions,
        x_monsters,
        x_event_options,
        x_character,
        x_rooms,
    ) = [x_block.contiguous() for x_block in x_tokens.split(widths, dim=1)]

    # L2 keys, once per token: its own class's net, the one every action type scores
    # it with. Monsters and the character are never L2 selections: zero keys
    x_keys = torch.cat(
        [
            model._pointer_key_card(x_cards),
            model._pointer_key_relic(x_relics),
            model._pointer_key_potion(x_potions),
            x_monsters.new_zeros(batch_size, x_monsters.shape[1], model._dim_key),
            model._pointer_key_event(x_event_options),
            x_character.new_zeros(batch_size, x_character.shape[1], model._dim_key),
            model._pointer_key_room(x_rooms),
        ],
        dim=1,
    )

    # L2: score the whole window, conditioned on the `ActionType` embedding;
    # the mask keeps exactly the type's legal tokens
    x_selects_token = model._selects_token[x_action_type]
    x_mask_l2 = x_legal[x_rows, x_action_type]

    # Every row at once: a type that selects no token (TurnEnd...) gets window position 0 as a
    # placeholder, so its categorical stays finite
    x_mask_l2[:, 0] |= ~x_selects_token

    # Recorded selections are (kind, rank); the pointer scores positions
    # A row without a selection records kind and rank -1: rank -1 hits no token, so _position
    # gives 0, the placeholder
    x_recorded_pos = None
    if x_recorded_kind is not None:
        x_recorded_pos = _position(x_slice_kind, x_recorded_kind, x_recorded_rank)

    # Calculate the window's logits & sample from them
    x_logits = model._head_level_2(
        x_keys, x_global, model._action_type_embedding(x_action_type), x_mask_l2
    )
    x_pos, x_log_prob_l2, x_entropy_l2, x_kl_uniform_l2 = _categorical_step(
        x_logits, x_mask_l2, x_recorded_pos, greedy, generator
    )
    x_kind, x_rank = _kind_rank(x_slice_kind, x_pos)

    # L3: monster target, only for actions that take one (CardPlay / PotionUse)
    # Keep only rows whose committed L2 selection takes a target: at recompute
    # the recorded selection says so (-1 = none), at sample time the mask does
    x_has_target = (
        x_recorded_target >= 0
        if x_recorded_target is not None
        else model._is_targeted[x_action_type] & x_needs_target[x_rows, x_pos]
    )

    # Score the monster block — the mask keeps the alive monsters — conditioned on the
    # selected card/potion token, plus a zero-key column: the other rows' placeholder,
    # appended, not column 0: out of combat the block has no columns
    x_mask_l3 = torch.cat(
        [x_monster_is_target & x_has_target.unsqueeze(1), ~x_has_target.unsqueeze(1)],
        dim=1,
    )
    x_selected_card_or_potion = x_tokens[x_rows, x_pos]

    # A cat, not F.pad: compiled, F.pad's backward fails on an empty monster block
    x_keys = torch.cat(
        [
            model._pointer_key_monster(x_monsters),
            x_monsters.new_zeros(batch_size, 1, model._dim_key),
        ],
        dim=1,
    )

    # The monster block is dense: a column is the monster's rank among the monster tokens
    x_recorded_pos_l3 = None
    if x_recorded_target is not None:
        x_recorded_pos_l3 = torch.where(x_has_target, x_recorded_target, x_monsters.shape[1])

    # Calculate the block's logits & sample from them
    x_logits = model._head_level_3(x_keys, x_global, x_selected_card_or_potion, x_mask_l3)
    x_target, x_log_prob_l3, x_entropy_l3, x_kl_uniform_l3 = _categorical_step(
        x_logits, x_mask_l3, x_recorded_pos_l3, greedy, generator
    )

    # A level a row doesn't reach: no selection (-1) and zero terms, whatever its placeholder drew
    x_kind = torch.where(x_selects_token, x_kind, -1)
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
        x_kind,
        x_rank,
        x_target,
        torch.stack([x_log_prob_l1, x_log_prob_l2, x_log_prob_l3], dim=1),
        torch.stack([x_entropy_l1, x_entropy_l2, x_entropy_l3], dim=1),
        torch.stack([x_kl_uniform_l1, x_kl_uniform_l2, x_kl_uniform_l3], dim=1),
    )


def compile_hot_paths(model: ActorCritic) -> None:
    """Fuse the update's hot paths: each transformer block, the draw-order layer's block, the map
    GNN's message passing, and the policy heads on recorded actions. Autotuned, so the first
    update compiles for a minute or two"""
    mode = "max-autotune-no-cudagraphs"
    for block in model.core._entity_transformer._entity_transformer_blocks:
        block.compile(dynamic=True, mode=mode)
    model.core._draw_order_layer._draw_order_block.compile(dynamic=True, mode=mode)
    model.core._map_gnn.message_passing = torch.compile(message_passing, dynamic=True, mode=mode)
    model.policy_heads_recorded = torch.compile(policy_heads, dynamic=True, mode=mode)
