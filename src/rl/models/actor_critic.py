import math

import torch
import torch.nn as nn

from src.rl.encoding.focus import ENCODING_DIM_FOCUS
from src.rl.models.core import Core
from src.rl.models.heads import HeadActionType
from src.rl.models.heads import HeadPointerSelect
from src.rl.models.heads import HeadValue
from src.rl.models.heads import PointerKeys
from src.rl.reward import REWARD_STREAMS
from src.rl.types import ACTION_TYPE_POOL
from src.rl.types import AT_MONSTER_TARGETED
from src.rl.types import NUM_ACTION_TYPES
from src.rl.types import SliceKind
from src.rl.types import X_Action
from src.rl.types import X_CoreOutput
from src.rl.types import X_GameState
from src.rl.types import X_Mask


# L2 pool kind -> the pointer-key net that scores the window for it. Every kind
# in ACTION_TYPE_POOL must appear (asserted at import)
_POOL_KEY_NET: dict[SliceKind, str] = {
    SliceKind.CARD_PILE_HAND: "_pointer_key_card",
    SliceKind.CARD_PILE_DRAW: "_pointer_key_card",
    SliceKind.CARD_PILE_DISCARD: "_pointer_key_card",
    SliceKind.CARD_PILE_EXHAUST: "_pointer_key_card",
    SliceKind.CARD_PILE_DECK: "_pointer_key_card",
    SliceKind.CARD_PILE_DISCOVER: "_pointer_key_card",
    SliceKind.CARD_REWARD_0: "_pointer_key_card",
    SliceKind.CARD_REWARD_1: "_pointer_key_card",
    SliceKind.CARD_REWARD_2: "_pointer_key_card",
    SliceKind.CARD_REWARD_3: "_pointer_key_card",
    SliceKind.CARD_REWARD_4: "_pointer_key_card",
    SliceKind.CARD_SHOP: "_pointer_key_card",
    SliceKind.CARD_PILE_EVENT_ROLL: "_pointer_key_card",
    SliceKind.RELIC_REWARD: "_pointer_key_relic",
    SliceKind.RELIC_SHOP: "_pointer_key_relic",
    SliceKind.RELIC_EVENT_ROLL: "_pointer_key_relic",
    SliceKind.POTION_OWNED: "_pointer_key_potion",
    SliceKind.POTION_REWARD: "_pointer_key_potion",
    SliceKind.POTION_SHOP: "_pointer_key_potion",
    SliceKind.POTION_EVENT_ROLL: "_pointer_key_potion",
    SliceKind.EVENT_OPTIONS: "_pointer_key_event",
    SliceKind.ROOMS: "_pointer_key_room",
}
# Barrier probability floor: each legal action's −log p contribution to the
# uniform-KL barrier caps at −log(1e-4) ≈ 9.2 nats
_KL_UNIFORM_LOG_P_FLOOR = math.log(1e-4)

_POOLED_KINDS: set = set().union(*ACTION_TYPE_POOL.values())
assert _POOLED_KINDS <= set(_POOL_KEY_NET), (
    "every L2 pool needs a pointer-key net: missing "
    f"{sorted(k.name for k in _POOLED_KINDS - set(_POOL_KEY_NET))}"
)
assert all(
    len({_POOL_KEY_NET[kind] for kind in kinds}) == 1 for kinds in ACTION_TYPE_POOL.values()
), "a multi-kind pool must share one pointer-key net"


class ActorCritic(nn.Module):
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
        checkpoint_privileged: bool = False,
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
            checkpoint_privileged=checkpoint_privileged,
        )

        # L1: one masked categorical over ActionType, focus-gated (GLU); a halt has an empty mask
        self._head_action_type = HeadActionType(
            self.core.dim_global,
            dim_ff_primary,
            num_choices=NUM_ACTION_TYPES,
            dim_context=ENCODING_DIM_FOCUS,
        )
        self._head_level_2 = HeadPointerSelect(self.core.dim_global, dim_op, dim_key)
        self._head_level_3 = HeadPointerSelect(self.core.dim_global, dim_entity, dim_key)
        self._head_value = HeadValue(self.core.dim_global, dim_ff_value, len(REWARD_STREAMS))

        # L2/L3 pointer selection: one key net per entity class + a shared query net per level
        self._action_type_embedding = nn.Embedding(NUM_ACTION_TYPES, dim_op)
        # One key net per entity class; every SliceKind of that class shares it, so an entity maps
        # to the same key in every selection context. _pointer_key_for routes a pool to its net
        self._pointer_key_card = PointerKeys(dim_entity, dim_key)
        self._pointer_key_relic = PointerKeys(dim_entity, dim_key)
        self._pointer_key_potion = PointerKeys(dim_entity, dim_key)
        self._pointer_key_event = PointerKeys(dim_entity, dim_key)
        self._pointer_key_room = PointerKeys(dim_entity, dim_key)
        self._pointer_key_monster = PointerKeys(dim_entity, dim_key)

        self._init_weights()

    def _init_weights(self) -> None:
        """PPO init: orthogonal (√2), near-zero logit layers (policy ≈ uniform), unit-gain value."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:  # Some projector layers are bias-free
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.MultiheadAttention):
                # in_proj is a raw Parameter, not a Linear — the isinstance walk
                # above misses it and left the Q/K/V projections at xavier
                nn.init.orthogonal_(m.in_proj_weight, gain=math.sqrt(2))
                nn.init.zeros_(m.in_proj_bias)
        nn.init.orthogonal_(self._head_action_type._scorer[-1].weight, gain=0.01)
        nn.init.orthogonal_(self._head_level_2._query_net[-1].weight, gain=0.01)
        nn.init.orthogonal_(self._head_level_3._query_net[-1].weight, gain=0.01)
        nn.init.orthogonal_(self._head_value._network[-1].weight, gain=1.0)

        # Resolve the pool -> key-net names once; getattr on nn.Module walks
        # _parameters/_buffers/_modules on every miss
        self._key_nets = {name: getattr(self, name) for name in set(_POOL_KEY_NET.values())}

    def _pointer_key_for(self, action_type: int) -> PointerKeys:
        """The key net for a policy action type, chosen by the entity class of
        its L2 pool. Table-driven off ACTION_TYPE_POOL so the virtual pending
        types route themselves — a pending pick over the draw pile scores card
        tokens, one over the staked relics scores relic tokens."""
        pools = ACTION_TYPE_POOL.get(action_type)
        name = _POOL_KEY_NET.get(pools[0]) if pools else None
        if name is None:
            raise ValueError(f"no pointer key net for policy action type {action_type}")
        return self._key_nets[name]

    def _categorical_step(
        self,
        x_logits: torch.Tensor,
        x_mask: torch.Tensor,
        x_recorded_idx: torch.Tensor | None = None,
        greedy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_masked_logits = x_logits.masked_fill(~x_mask, float("-inf"))
        dist = torch.distributions.Categorical(logits=x_masked_logits)

        # Determine which index to sample
        if x_recorded_idx is not None:
            # PPO recompute path
            x_idx = x_recorded_idx
        elif greedy:
            # Highest logit
            x_idx = torch.argmax(x_masked_logits, dim=-1)
        else:
            # Sample from distribution
            x_idx = dist.sample()

        # KL(uniform‖π) over the legal set — the anti-collapse barrier. Entropy's
        # restoring force on a dying action vanishes with its probability (∝ p·|log p|);
        # this gradient tends to a constant ~1/n instead, flooring legal-action probs
        # near coef/(n·|A|) (log-barrier regularization, Agarwal et al. 2021).
        # −log p is clamped at the floor: unbounded, the barrier dominated the loss
        # once a head neared determinism (EFFECT-I L2 at 153; IIIc L3 at 10 with
        # dead-target −log p ≈ 21) while grad clipping neutered its restoring force.
        # Below ε the action is conceded and the loss stays bounded
        x_n_legal = x_mask.sum(-1).clamp(min=1).float()  # min=1: empty/singleton rows → KL 0
        x_logp = dist.logits.clamp(min=_KL_UNIFORM_LOG_P_FLOOR)
        x_kl_uniform = -x_logp.masked_fill(~x_mask, 0.0).sum(-1) / x_n_legal - x_n_legal.log()

        return x_idx, dist.log_prob(x_idx), dist.entropy(), x_kl_uniform

    def _compute_action_type(
        self,
        x_game_state: X_GameState,
        x_core_out: X_CoreOutput,
        x_mask: X_Mask,
        greedy: bool,
        x_recorded: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pick the `ActionType` for every row — one masked categorical, focus-gated."""
        x_logits = self._head_action_type(
            x_core_out.global_, x_game_state.focus, x_mask.mask_action_type
        )
        return self._categorical_step(x_logits, x_mask.mask_action_type, x_recorded, greedy)

    def _compute_action_idxs(
        self,
        x_idx_at: torch.Tensor,
        x_game_state: X_GameState,
        x_core_out: X_CoreOutput,
        x_mask: X_Mask,
        greedy: bool,
        x_recorded_kind: torch.Tensor | None = None,
        x_recorded_rank: torch.Tensor | None = None,
        x_recorded_target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        batch_size = x_idx_at.shape[0]
        device = x_idx_at.device
        x_idxs = x_core_out.entities.x_slice_kind

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
        x_legal = torch.cat([block.legal for block in blocks], dim=-1)
        x_needs_target = torch.cat([block.needs_target for block in blocks], dim=-1)
        x_is_target = torch.cat([block.is_target for block in blocks], dim=-1)

        # Initialize tensors to store the picks (engine-facing: kind + rank) and stats
        x_kind = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        x_rank = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        x_log_prob_l1 = torch.zeros(batch_size, device=device)
        x_entropy_l1 = torch.zeros(batch_size, device=device)
        x_kl_uniform_l1 = torch.zeros(batch_size, device=device)
        x_target = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        x_log_prob_l2 = torch.zeros(batch_size, device=device)
        x_entropy_l2 = torch.zeros(batch_size, device=device)
        x_kl_uniform_l2 = torch.zeros(batch_size, device=device)

        for action_type in x_idx_at.unique().tolist():
            if action_type not in ACTION_TYPE_POOL:  # terminal type — no index
                continue

            # Selection mask keeping samples for the current `ActionType` in the loop
            x_rows = (x_idx_at == action_type).nonzero(as_tuple=True)[0]
            x_tokens = x_core_out.entities.x[x_rows]
            x_idxs_rows = x_idxs[x_rows]

            # Level 1: score the whole window, gated by the `ActionType` embedding;
            # the mask keeps exactly the type's pool
            x_mask_l1 = x_legal[x_rows, action_type]
            x_keys = self._pointer_key_for(action_type)(x_tokens)
            x_action_type_emb = self._action_type_embedding(torch.full_like(x_rows, action_type))

            # Recorded picks are (kind, rank); the pointer scores positions
            x_recorded_pos = None
            if x_recorded_kind is not None:
                x_recorded_pos = self._position(
                    x_idxs_rows, x_recorded_kind[x_rows], x_recorded_rank[x_rows]
                )

            # Calculate the window's logits & sample from them
            x_logits = self._head_level_2(
                x_keys, x_core_out.global_[x_rows], x_action_type_emb, x_mask_l1
            )
            (
                x_pos,
                x_log_prob_l1[x_rows],
                x_entropy_l1[x_rows],
                x_kl_uniform_l1[x_rows],
            ) = self._categorical_step(x_logits, x_mask_l1, x_recorded_pos, greedy)
            x_kind[x_rows], x_rank[x_rows] = self._kind_rank(x_idxs_rows, x_pos)

            # Level 2: monster target, only for actions that take one (CardPlay / PotionUse)
            if action_type not in AT_MONSTER_TARGETED:
                continue

            # Keep only rows whose committed L1 selection takes a target: at recompute
            # the recorded pick says so (-1 = none), at sample time the mask does
            x_recorded_rows = self._rows(x_recorded_target, x_rows)
            x_has_target = (
                x_recorded_rows >= 0
                if x_recorded_rows is not None
                else x_needs_target[x_rows, x_pos]
            )
            x_local = torch.nonzero(x_has_target, as_tuple=True)[0]
            if x_local.numel() == 0:
                continue

            # Subset of `ActionType`-specifc samples that also require targeting
            x_rows_l2 = x_rows[x_local]
            x_idxs_l2 = x_idxs_rows[x_local]

            # Score the whole window like L2 — the mask keeps the alive monster
            # tokens — conditioned on the selected card/potion token
            x_mask_l2 = x_is_target[x_rows_l2]
            x_selected_card_or_pot = x_tokens[x_local, x_pos[x_local]]
            x_keys = self._pointer_key_monster(x_tokens[x_local])
            x_logits = self._head_level_3(
                x_keys, x_core_out.global_[x_rows_l2], x_selected_card_or_pot, x_mask_l2
            )
            x_recorded_local = None
            if x_recorded_rows is not None:
                x_monsters = torch.full_like(x_local, SliceKind.MONSTERS)
                x_recorded_local = self._position(x_idxs_l2, x_monsters, x_recorded_rows[x_local])
            (
                x_pos_l2,
                x_log_prob_l2[x_rows_l2],
                x_entropy_l2[x_rows_l2],
                x_kl_uniform_l2[x_rows_l2],
            ) = self._categorical_step(x_logits, x_mask_l2, x_recorded_local, greedy)
            # The target is the monster's rank among the monster tokens = its engine index
            x_target[x_rows_l2] = self._kind_rank(x_idxs_l2, x_pos_l2)[1]

        return (
            x_kind,
            x_rank,
            x_log_prob_l1,
            x_entropy_l1,
            x_kl_uniform_l1,
            x_target,
            x_log_prob_l2,
            x_entropy_l2,
            x_kl_uniform_l2,
        )

    @staticmethod
    def _position(
        x_idxs: torch.Tensor, x_kind: torch.Tensor, x_rank: torch.Tensor
    ) -> torch.Tensor:
        """The window position of the rank-th token of `kind`, per row."""
        x_match = x_idxs == x_kind.unsqueeze(1)
        x_hit = x_match & (x_match.cumsum(1) == x_rank.unsqueeze(1) + 1)
        return x_hit.long().argmax(1)

    @staticmethod
    def _kind_rank(x_idxs: torch.Tensor, x_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """A window position's kind and its rank within that kind, per row."""
        x_kind = x_idxs.gather(1, x_pos.unsqueeze(1)).squeeze(1)
        x_before = torch.arange(x_idxs.shape[1], device=x_idxs.device) < x_pos.unsqueeze(1)
        return x_kind, ((x_idxs == x_kind.unsqueeze(1)) & x_before).sum(1)

    @staticmethod
    def _rows(x_recorded: torch.Tensor | None, x_rows: torch.Tensor) -> torch.Tensor | None:
        """Recorded indices for `x_rows`, or None when nothing was recorded (sampling)."""
        return None if x_recorded is None else x_recorded[x_rows]

    def _run(
        self,
        x_game_state: X_GameState,
        x_core_out: X_CoreOutput,
        x_mask: X_Mask,
        greedy: bool,
        x_recorded_action_type: torch.Tensor | None = None,
        x_recorded_kind: torch.Tensor | None = None,
        x_recorded_rank: torch.Tensor | None = None,
        x_recorded_target: torch.Tensor | None = None,
    ) -> X_Action:
        # Compute `ActionType` first
        x_idx_at, x_log_prob_at, x_entropy_at, x_kl_uniform_at = self._compute_action_type(
            x_game_state, x_core_out, x_mask, greedy, x_recorded_action_type
        )

        # Compute the `Action`'s picks from the chosen `ActionType`
        (
            x_kind,
            x_rank,
            x_log_prob_l1,
            x_entropy_l1,
            x_kl_uniform_l1,
            x_target,
            x_log_prob_l2,
            x_entropy_l2,
            x_kl_uniform_l2,
        ) = self._compute_action_idxs(
            x_idx_at,
            x_game_state,
            x_core_out,
            x_mask,
            greedy,
            x_recorded_kind,
            x_recorded_rank,
            x_recorded_target,
        )

        # Stack per-level columns: contiguous, faster than strided writes to a preallocated buffer
        return X_Action(
            action_type=x_idx_at,
            kind=x_kind,
            rank=x_rank,
            target=x_target,
            log_prob=torch.stack([x_log_prob_at, x_log_prob_l1, x_log_prob_l2], dim=1),
            entropy=torch.stack([x_entropy_at, x_entropy_l1, x_entropy_l2], dim=1),
            kl_uniform=torch.stack([x_kl_uniform_at, x_kl_uniform_l1, x_kl_uniform_l2], dim=1),
            batch_size=[x_idx_at.shape[0]],
        )

    def forward(
        self, x_game_state: X_GameState, x_mask: X_Mask, greedy: bool = False
    ) -> tuple[X_Action, torch.Tensor]:
        x_core_out = self.core(x_game_state)
        x_values = self._head_value(x_core_out.global_value)
        return self._run(x_game_state, x_core_out, x_mask, greedy), x_values

    def evaluate_actions(
        self,
        x_game_state: X_GameState,
        x_mask: X_Mask,
        x_action_type: torch.Tensor,
        x_kind: torch.Tensor,
        x_rank: torch.Tensor,
        x_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_core_out = self.core(x_game_state)
        x_values = self._head_value(x_core_out.global_value)
        x_actions = self._run(
            x_game_state,
            x_core_out,
            x_mask,
            greedy=True,
            x_recorded_action_type=x_action_type,
            x_recorded_kind=x_kind,
            x_recorded_rank=x_rank,
            x_recorded_target=x_target,
        )
        return x_actions.log_prob.sum(-1), x_actions.entropy, x_actions.kl_uniform, x_values
