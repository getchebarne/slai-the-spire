import math

import torch
import torch.nn as nn
from slai import ActionType

from src.rl.encoding.screen import ENCODING_DIM_SCREEN
from src.rl.encoding.shop import ENCODING_DIM_PRICE
from src.rl.models.core import Core
from src.rl.models.heads import HeadActionType
from src.rl.models.heads import HeadPointerSelect
from src.rl.models.heads import HeadValue
from src.rl.models.heads import PointerKeys
from src.rl.reward import REWARD_STREAMS
from src.rl.types import ACTION_TYPE_BY_INT
from src.rl.types import ACTION_TYPE_POOL
from src.rl.types import NUM_ACTION_TYPES
from src.rl.types import SliceKind
from src.rl.types import TAction
from src.rl.types import TCoreOutput
from src.rl.types import TGameState
from src.rl.types import TMask


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

        # L1: one masked categorical over ActionType, screen-gated (GLU); a halt has an empty mask.
        self._head_action_type = HeadActionType(
            self.core.dim_global,
            dim_ff_primary,
            num_choices=NUM_ACTION_TYPES,
            dim_context=ENCODING_DIM_SCREEN,
        )
        self._head_level_2 = HeadPointerSelect(self.core.dim_global, dim_op, dim_key)
        self._head_level_3 = HeadPointerSelect(self.core.dim_global, dim_entity, dim_key)
        self._head_value = HeadValue(self.core.dim_global, dim_ff_value, len(REWARD_STREAMS))

        # L2/L3 pointer selection: one key net per entity class + a shared query net per level.
        self._action_type_embedding = nn.Embedding(NUM_ACTION_TYPES, dim_op)
        # One key net per entity class; every SliceKind of that class shares it, so an entity maps
        # to the same key in every selection context. _pointer_key_for routes a pool to its net.
        self._pointer_key_card = PointerKeys(dim_entity, dim_key)
        self._pointer_key_relic = PointerKeys(dim_entity, dim_key)
        self._pointer_key_potion = PointerKeys(dim_entity, dim_key)
        self._pointer_key_event = PointerKeys(dim_entity, dim_key)
        self._pointer_key_room = PointerKeys(dim_entity, dim_key)
        self._pointer_key_monster = PointerKeys(dim_entity, dim_key)

        self._shop_price_keys = nn.Linear(ENCODING_DIM_PRICE, dim_key, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """PPO init: orthogonal (√2), near-zero logit layers (policy ≈ uniform), unit-gain value."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:  # _shop_price_keys is bias-free
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self._head_action_type._scorer[-1].weight, gain=0.01)
        nn.init.orthogonal_(self._head_level_2._query_net[-1].weight, gain=0.01)
        nn.init.orthogonal_(self._head_level_3._query_net[-1].weight, gain=0.01)
        nn.init.orthogonal_(self._head_value._network[-1].weight, gain=1.0)

    def _pointer_key_for(self, action_type: int) -> PointerKeys:
        match ACTION_TYPE_BY_INT[action_type]:
            case (
                ActionType.CardDiscard
                | ActionType.CardDiscover
                | ActionType.CardDuplicate
                | ActionType.CardNightmare
                | ActionType.CardPlay
                | ActionType.CardPurge
                | ActionType.CardRetain
                | ActionType.CardSetup
                | ActionType.CardTransform
                | ActionType.CardUpgrade
                | ActionType.RewardTakeCard
                | ActionType.ShopBuyCard
                | ActionType.ShopPurge
            ):
                return self._pointer_key_card
            case ActionType.ShopBuyRelic:
                return self._pointer_key_relic
            case ActionType.PotionDiscard | ActionType.PotionUse | ActionType.ShopBuyPotion:
                return self._pointer_key_potion
            case ActionType.EventOptionSelect:
                return self._pointer_key_event
            case ActionType.RoomSelect:
                return self._pointer_key_room
        raise ValueError(f"no pointer key net for {ACTION_TYPE_BY_INT[action_type]}")

    def _categorical_step(
        self,
        t_logits: torch.Tensor,
        t_mask: torch.Tensor,
        t_recorded_idx: torch.Tensor | None = None,
        greedy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_masked_logits = t_logits.masked_fill(~t_mask, float("-inf"))
        dist = torch.distributions.Categorical(logits=t_masked_logits)

        # Determine which index to sample
        if t_recorded_idx is not None:
            # PPO recompute path
            t_idx = t_recorded_idx
        elif greedy:
            # Highest logit
            t_idx = torch.argmax(t_masked_logits, dim=-1)
        else:
            # Sample from distribution
            t_idx = dist.sample()

        return t_idx, dist.log_prob(t_idx), dist.entropy()

    def _compute_action_type(
        self,
        t_game_state: TGameState,
        t_core_out: TCoreOutput,
        t_mask: TMask,
        greedy: bool,
        t_recorded: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pick the `ActionType` for every row — one masked categorical, screen-gated."""
        t_logits = self._head_action_type(
            t_core_out.global_, t_game_state.screen, t_mask.mask_action_type
        )
        return self._categorical_step(t_logits, t_mask.mask_action_type, t_recorded, greedy)

    def _compute_action_idxs(
        self,
        t_idx_at: torch.Tensor,
        t_game_state: TGameState,
        t_core_out: TCoreOutput,
        t_mask: TMask,
        greedy: bool,
        t_recorded_l1: torch.Tensor | None = None,
        t_recorded_l2: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        batch_size = t_idx_at.shape[0]
        device = t_idx_at.device

        # Initialize tensors to store index results
        t_idx_l1 = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        t_log_prob_l1 = torch.zeros(batch_size, device=device)
        t_entropy_l1 = torch.zeros(batch_size, device=device)
        t_idx_l2 = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        t_log_prob_l2 = torch.zeros(batch_size, device=device)
        t_entropy_l2 = torch.zeros(batch_size, device=device)

        for action_type in t_idx_at.unique().tolist():
            pool = ACTION_TYPE_POOL.get(ACTION_TYPE_BY_INT[action_type])
            if pool is None:  # terminal type — no index
                continue

            # Selection mask keeping samples for the current `ActionType` in the loop
            t_rows = (t_idx_at == action_type).nonzero(as_tuple=True)[0]

            # Level 1: score the action's pool, gated by its `ActionType` embedding
            t_mask_l1 = t_mask.mask_action_idx[str(action_type)][t_rows]
            t_keys = self._pointer_key_for(action_type)(t_core_out.pool[pool][t_rows])
            t_action_type_emb = self._action_type_embedding(torch.full_like(t_rows, action_type))

            # Add price key iff we're scoring a shop purchase
            t_prices = _get_shop_prices_for(pool, t_game_state)
            if t_prices is not None:  # shop purchase: additive price-key term
                t_keys = t_keys + self._shop_price_keys(t_prices[t_rows])

            # Calculate pool's logits & sample from them
            t_logits = self._head_level_2(
                t_keys, t_core_out.global_[t_rows], t_action_type_emb, t_mask_l1
            )
            t_idx_l1[t_rows], t_log_prob_l1[t_rows], t_entropy_l1[t_rows] = self._categorical_step(
                t_logits, t_mask_l1, self._rows(t_recorded_l1, t_rows), greedy
            )

            # Level 2: monster target, only for actions that take one (CardPlay / PotionUse)
            t_target_mask = _get_target_mask_for(action_type, t_mask)
            if t_target_mask is None:
                continue

            # Keep only rows whose committed L1 selection has a legal target
            t_l1 = t_idx_l1[t_rows]
            t_monster_mask = t_target_mask[t_rows, t_l1]
            t_recorded_rows = self._rows(t_recorded_l2, t_rows)
            t_has_target = (
                t_recorded_rows >= 0 if t_recorded_rows is not None else t_monster_mask.any(dim=-1)
            )
            t_local = torch.nonzero(t_has_target, as_tuple=True)[0]
            if t_local.numel() == 0:
                continue

            # Subset of `ActionType`-specifc samples that also require targeting
            t_rows_l2 = t_rows[t_local]

            # Score monsters for the surviving rows, conditioned on the selected entity
            t_mask_l2 = t_monster_mask[t_local]
            t_selected_card_or_pot = t_core_out.pool[pool][t_rows_l2, t_l1[t_local]]
            t_keys = self._pointer_key_monster(t_core_out.pool[SliceKind.MONSTERS][t_rows_l2])
            t_logits = self._head_level_3(
                t_keys, t_core_out.global_[t_rows_l2], t_selected_card_or_pot, t_mask_l2
            )
            t_recorded_local = t_recorded_rows[t_local] if t_recorded_rows is not None else None
            t_idx_l2[t_rows_l2], t_log_prob_l2[t_rows_l2], t_entropy_l2[t_rows_l2] = (
                self._categorical_step(t_logits, t_mask_l2, t_recorded_local, greedy)
            )

        return (
            t_idx_l1,
            t_log_prob_l1,
            t_entropy_l1,
            t_idx_l2,
            t_log_prob_l2,
            t_entropy_l2,
        )

    @staticmethod
    def _rows(t_recorded: torch.Tensor | None, t_rows: torch.Tensor) -> torch.Tensor | None:
        """Recorded indices for `t_rows`, or None when nothing was recorded (sampling)."""
        return None if t_recorded is None else t_recorded[t_rows]

    def _run(
        self,
        t_game_state: TGameState,
        t_core_out: TCoreOutput,
        t_mask: TMask,
        greedy: bool,
        t_recorded_action_type: torch.Tensor | None = None,
        t_recorded_l1: torch.Tensor | None = None,
        t_recorded_l2: torch.Tensor | None = None,
    ) -> TAction:
        # Compute `ActionType` first
        t_idx_at, t_log_prob_at, t_entropy_at = self._compute_action_type(
            t_game_state, t_core_out, t_mask, greedy, t_recorded_action_type
        )

        # Compute the `Action`'s indexes from the chosen `ActionType`
        (
            t_idx_l1,
            t_log_prob_l1,
            t_entropy_l1,
            t_idx_l2,
            t_log_prob_l2,
            t_entropy_l2,
        ) = self._compute_action_idxs(
            t_idx_at, t_game_state, t_core_out, t_mask, greedy, t_recorded_l1, t_recorded_l2
        )

        # Stack per-level columns: contiguous, faster than strided writes to a preallocated buffer
        return TAction(
            idxs=torch.stack([t_idx_at, t_idx_l1, t_idx_l2], dim=1),
            log_prob=torch.stack([t_log_prob_at, t_log_prob_l1, t_log_prob_l2], dim=1),
            entropy=torch.stack([t_entropy_at, t_entropy_l1, t_entropy_l2], dim=1),
            batch_size=[t_idx_at.shape[0]],
        )

    def forward(
        self, t_game_state: TGameState, t_mask: TMask, greedy: bool = False
    ) -> tuple[TAction, torch.Tensor]:
        t_core_out = self.core(t_game_state)
        t_values = self._head_value(t_core_out.global_)
        return self._run(t_game_state, t_core_out, t_mask, greedy), t_values

    def evaluate_actions(
        self,
        t_game_state: TGameState,
        t_mask: TMask,
        t_idx_at: torch.Tensor,
        t_idx_l1: torch.Tensor,
        t_idx_l2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_core_out = self.core(t_game_state)
        t_values = self._head_value(t_core_out.global_)
        t_actions = self._run(
            t_game_state,
            t_core_out,
            t_mask,
            greedy=True,
            t_recorded_action_type=t_idx_at,
            t_recorded_l1=t_idx_l1,
            t_recorded_l2=t_idx_l2,
        )
        return t_actions.log_prob.sum(-1), t_actions.entropy, t_values


def _get_target_mask_for(action_type: int, t_mask: TMask) -> torch.Tensor | None:
    """The L3 monster-target mask for an action's selection pool, or None if it takes no target."""
    match ACTION_TYPE_BY_INT[action_type]:
        case ActionType.CardPlay:
            return t_mask.mask_target_card
        case ActionType.PotionUse:
            return t_mask.mask_target_potion

    return None


def _get_shop_prices_for(pool: SliceKind, t_game_state: TGameState) -> torch.Tensor | None:
    """Per-item shop prices for a shop selection pool, or None if the pool isn't a shop."""
    match pool:
        case SliceKind.CARD_SHOP:
            return t_game_state.shop_card_prices
        case SliceKind.RELIC_SHOP:
            return t_game_state.shop_relic_prices
        case SliceKind.POTION_SHOP:
            return t_game_state.shop_potion_prices

    return None
