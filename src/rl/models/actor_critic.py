"""
Actor-Critic model with per-FSM-state hierarchical action heads.

Architecture (post-migration):
1. Core encoder processes game state → entity embeddings + global context.
2. Samples are routed by HeadTypePrimary (determined by slai.Phase).
3. Decision primaries (COMBAT_DEFAULT, CARD_REWARD, REST_SITE,
   RELIC_REWARD): a binary head decides skip/take, then a secondary head
   picks the entity if "take".
4. Direct primaries (COMBAT_CARD_DISCARD, COMBAT_AWAIT_RETAIN,
   COMBAT_AWAIT_NIGHTMARE, COMBAT_AWAIT_SETUP, MAP_SELECT): the
   selection head fires immediately.
5. **Inline targeting**: when COMBAT_DEFAULT picks "play_card" and the
   chosen card has `requires_target`, the same forward pass also runs
   HeadMonsterSelect to pick the monster. The output bundles `target_idx`.
6. **Multi-pick retain**: COMBAT_AWAIT_RETAIN runs HeadCardRetain which
   sequentially samples `num` distinct hand indices.
7. Value head estimates state value.
"""

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn

from src.rl.action_space.masks import MaskBatch
from src.rl.action_space.types import HeadTypePrimary
from src.rl.action_space.types import IS_DECISION_PRIMARY
from src.rl.action_space.types import NUM_PRIMARY_HEADS
from src.rl.action_space.types import to_action
from src.rl.constants import MAP_WIDTH
from src.rl.constants import MAX_RELIC_REWARDS
from src.rl.constants import MAX_SIZE_HAND
from src.rl.encoding.state import XGameState
from src.rl.models.core import Core
from src.rl.models.core import CoreOutput
from src.rl.models.heads import HeadBinaryChoice
from src.rl.models.heads import HeadCardMultiPick
from src.rl.models.heads import HeadCardNightmare
from src.rl.models.heads import HeadCardPlay
from src.rl.models.heads import HeadCardRewardSelect
from src.rl.models.heads import HeadCardSetup
from src.rl.models.heads import HeadCardUpgrade
from src.rl.models.heads import HeadMapSelect
from src.rl.models.heads import HeadMonsterSelect
from src.rl.models.heads import HeadRelicSelect
from src.rl.models.heads import HeadValue


# =============================================================================
# Output Types
# =============================================================================


class ForwardOutput(NamedTuple):
    """
    Output from batched forward pass. All tensors are full-batch (B,) /
    (B, K) / (B, 1).
    """

    # Which primary group each sample belongs to
    head_type_primaries: torch.Tensor  # (B,) int64

    # Primary decision (-1 for direct primaries, 0=skip/end, 1=take/play)
    primary_indices: torch.Tensor  # (B,) int64
    primary_log_probs: torch.Tensor  # (B,)

    # Single entity selection (-1 if terminal e.g. end_turn/skip/rest)
    selection_indices: torch.Tensor  # (B,) int64
    selection_log_probs: torch.Tensor  # (B,)

    # Inline target for COMBAT_DEFAULT play-card-with-target (-1 otherwise)
    target_indices: torch.Tensor  # (B,) int64
    target_log_probs: torch.Tensor  # (B,)

    # Multi-pick output for COMBAT_AWAIT_RETAIN (-1 in unused slots)
    retain_indices: torch.Tensor  # (B, MAX_SIZE_HAND) int64
    retain_log_probs: torch.Tensor  # (B,)

    # Value estimate
    values: torch.Tensor  # (B, 1)

    def get_action(self, idx: int):
        """Convert sample at index to a slai action."""
        htp = HeadTypePrimary(self.head_type_primaries[idx].item())
        pi = self.primary_indices[idx].item()
        si = self.selection_indices[idx].item()
        ti = self.target_indices[idx].item()
        ri_list = [int(x) for x in self.retain_indices[idx].tolist() if x >= 0]
        return to_action(
            htp, pi, si,
            target_index=ti,
            retain_indices=ri_list,
        )

    def get_log_prob(self, idx: int) -> torch.Tensor:
        """Total log prob = primary + selection + target + retain (zeros for unused)."""
        return (
            self.primary_log_probs[idx]
            + self.selection_log_probs[idx]
            + self.target_log_probs[idx]
            + self.retain_log_probs[idx]
        )


@dataclass
class SingleOutput:
    """Convenience wrapper for single-sample inference."""

    head_type_primary: HeadTypePrimary
    primary_index: int
    primary_log_prob: torch.Tensor
    selection_index: int
    selection_log_prob: torch.Tensor
    target_index: int
    target_log_prob: torch.Tensor
    retain_indices: list[int]
    retain_log_prob: torch.Tensor
    value: torch.Tensor

    def to_action(self):
        return to_action(
            self.head_type_primary,
            self.primary_index,
            self.selection_index,
            target_index=self.target_index,
            retain_indices=self.retain_indices,
        )

    @property
    def log_prob(self) -> torch.Tensor:
        return (
            self.primary_log_prob
            + self.selection_log_prob
            + self.target_log_prob
            + self.retain_log_prob
        )


# =============================================================================
# Helpers
# =============================================================================

# Cached HTP int values (avoid attribute lookup in hot loops)
_HTP_CARD_REWARD = int(HeadTypePrimary.CARD_REWARD)
_HTP_COMBAT_CARD_DISCARD = int(HeadTypePrimary.COMBAT_CARD_DISCARD)
_HTP_COMBAT_DEFAULT = int(HeadTypePrimary.COMBAT_DEFAULT)
_HTP_MAP_SELECT = int(HeadTypePrimary.MAP_SELECT)
_HTP_REST_SITE = int(HeadTypePrimary.REST_SITE)
_HTP_COMBAT_AWAIT_RETAIN = int(HeadTypePrimary.COMBAT_AWAIT_RETAIN)
_HTP_COMBAT_AWAIT_NIGHTMARE = int(HeadTypePrimary.COMBAT_AWAIT_NIGHTMARE)
_HTP_COMBAT_AWAIT_SETUP = int(HeadTypePrimary.COMBAT_AWAIT_SETUP)
_HTP_RELIC_REWARD = int(HeadTypePrimary.RELIC_REWARD)


def _build_entity_tensors(core_out: CoreOutput) -> list[torch.Tensor | None]:
    """Per-head-type entity tensor (None where the head doesn't consume one)."""
    et: list[torch.Tensor | None] = [None] * NUM_PRIMARY_HEADS
    et[_HTP_CARD_REWARD] = core_out.x_combat_reward
    et[_HTP_COMBAT_DEFAULT] = core_out.x_hand
    et[_HTP_MAP_SELECT] = core_out.x_map
    et[_HTP_REST_SITE] = core_out.x_deck
    et[_HTP_COMBAT_AWAIT_RETAIN] = core_out.x_hand
    et[_HTP_COMBAT_CARD_DISCARD] = core_out.x_hand
    et[_HTP_COMBAT_AWAIT_NIGHTMARE] = core_out.x_hand
    et[_HTP_COMBAT_AWAIT_SETUP] = core_out.x_hand
    # _HTP_RELIC_REWARD: HeadRelicSelect uses only x_global; entity tensor unused.
    return et


# =============================================================================
# Model
# =============================================================================


class ActorCritic(nn.Module):

    def __init__(
        self,
        dim_entity: int = 128,
        dim_global: int = 256,
        transformer_dim_ff: int = 256,
        transformer_num_heads: int = 4,
        transformer_num_blocks: int = 2,
        map_encoder_kernel_size: int = 3,
        map_encoder_dim: int = 32,
        dim_ff_primary: int = 128,
        dim_ff_card: int = 128,
        dim_ff_monster: int = 128,
        dim_ff_map: int = 128,
        dim_ff_value: int = 128,
    ):
        super().__init__()

        self.core = Core(
            dim_entity=dim_entity,
            dim_global=dim_global,
            transformer_dim_ff=transformer_dim_ff,
            transformer_num_heads=transformer_num_heads,
            transformer_num_blocks=transformer_num_blocks,
            map_encoder_kernel_size=map_encoder_kernel_size,
            map_encoder_dim=map_encoder_dim,
        )

        dim_global = self.core.dim_global
        dim_map = self.core.dim_map

        # Decision primary heads (binary [skip, take])
        self.head_combat_default = HeadBinaryChoice(dim_global, dim_ff_primary)
        self.head_card_reward = HeadBinaryChoice(dim_global, dim_ff_primary)
        self.head_rest_site = HeadBinaryChoice(dim_global, dim_ff_primary)
        self.head_relic_reward_decide = HeadBinaryChoice(dim_global, dim_ff_primary)

        # Entity selection heads
        self.head_card_play = HeadCardPlay(dim_entity, dim_global, dim_ff_card)
        self.head_card_reward_select = HeadCardRewardSelect(dim_entity, dim_global, dim_ff_card)
        self.head_card_upgrade = HeadCardUpgrade(dim_entity, dim_global, dim_ff_card)
        self.head_card_setup = HeadCardSetup(dim_entity, dim_global, dim_ff_card)
        self.head_card_nightmare = HeadCardNightmare(dim_entity, dim_global, dim_ff_card)
        self.head_monster_select = HeadMonsterSelect(dim_entity, dim_global, dim_ff_monster)
        self.head_map_select = HeadMapSelect(dim_map, dim_global, dim_ff_map, MAP_WIDTH)
        self.head_relic_select = HeadRelicSelect(dim_global, dim_ff_card, MAX_RELIC_REWARDS)
        # Multi-pick heads (separate parameters for retain vs discard).
        self.head_card_retain = HeadCardMultiPick(dim_entity, dim_global, dim_ff_card)
        self.head_card_discard_multi = HeadCardMultiPick(dim_entity, dim_global, dim_ff_card)

        # Value head
        self.head_value = HeadValue(dim_global, dim_ff_value)

        # ---- Registries: list-indexed by int(HeadTypePrimary) ----

        self._decision_heads: list[HeadBinaryChoice | None] = [None] * NUM_PRIMARY_HEADS
        self._decision_heads[_HTP_COMBAT_DEFAULT] = self.head_combat_default
        self._decision_heads[_HTP_CARD_REWARD] = self.head_card_reward
        self._decision_heads[_HTP_REST_SITE] = self.head_rest_site
        self._decision_heads[_HTP_RELIC_REWARD] = self.head_relic_reward_decide

        self._selection_heads: list[nn.Module | None] = [None] * NUM_PRIMARY_HEADS
        self._selection_heads[_HTP_COMBAT_DEFAULT] = self.head_card_play
        self._selection_heads[_HTP_CARD_REWARD] = self.head_card_reward_select
        self._selection_heads[_HTP_REST_SITE] = self.head_card_upgrade
        self._selection_heads[_HTP_COMBAT_AWAIT_NIGHTMARE] = self.head_card_nightmare
        self._selection_heads[_HTP_COMBAT_AWAIT_SETUP] = self.head_card_setup
        self._selection_heads[_HTP_MAP_SELECT] = self.head_map_select
        self._selection_heads[_HTP_RELIC_REWARD] = self.head_relic_select
        # COMBAT_AWAIT_RETAIN and COMBAT_CARD_DISCARD use multi-pick heads
        # (head_card_retain / head_card_discard_multi) via dedicated dispatch.

    def forward(
        self,
        x_game_state: XGameState,
        mask_batch: MaskBatch,
        sample: bool = True,
    ) -> ForwardOutput:
        device = x_game_state.x_hand.device

        # 1. Core encoder
        core_out = self.core(x_game_state)
        B = core_out.x_global.shape[0]

        # 2. Value head
        values = self.head_value(core_out.x_global)

        # 3. Initialize output tensors
        head_type_primaries = torch.full((B,), -1, dtype=torch.long, device=device)
        primary_indices = torch.full((B,), -1, dtype=torch.long, device=device)
        primary_log_probs = torch.zeros(B, device=device)
        selection_indices = torch.full((B,), -1, dtype=torch.long, device=device)
        selection_log_probs = torch.zeros(B, device=device)
        target_indices = torch.full((B,), -1, dtype=torch.long, device=device)
        target_log_probs = torch.zeros(B, device=device)
        retain_indices = torch.full((B, MAX_SIZE_HAND), -1, dtype=torch.long, device=device)
        retain_log_probs = torch.zeros(B, device=device)

        # 4. Pre-extract entity tensors
        entity_tensors = _build_entity_tensors(core_out)

        # 5. Process each primary group
        for htp in range(NUM_PRIMARY_HEADS):
            idx = mask_batch.route[htp]
            if len(idx) == 0:
                continue

            head_type_primaries[idx] = htp
            x_global_group = core_out.x_global[idx]

            # ---- Multi-pick (retain + discard share machinery) ----
            if htp == _HTP_COMBAT_AWAIT_RETAIN or htp == _HTP_COMBAT_CARD_DISCARD:
                entities_group = core_out.x_hand[idx]
                sel_mask = mask_batch.selection_masks[htp]
                nums = mask_batch.retain_nums[idx]
                multi_head = (
                    self.head_card_retain
                    if htp == _HTP_COMBAT_AWAIT_RETAIN
                    else self.head_card_discard_multi
                )
                mp_out = multi_head(
                    entities_group, x_global_group, sel_mask, nums, sample=sample
                )
                retain_indices[idx] = mp_out.indices
                retain_log_probs[idx] = mp_out.log_prob
                continue

            if IS_DECISION_PRIMARY[htp]:
                # ---- Decision primary: binary head, then optional secondary ----
                primary_mask = mask_batch.primary_masks[htp]
                decision_head = self._decision_heads[htp]
                out = decision_head(x_global_group, primary_mask, sample)

                if sample:
                    chosen = out.indices
                    primary_indices[idx] = chosen
                    primary_log_probs[idx] = out.log_probs
                else:
                    chosen = torch.argmax(out.logits, dim=-1)
                    primary_indices[idx] = chosen

                # Run secondary head for samples that chose "select" (idx==1)
                needs_secondary = chosen == 1
                if torch.any(needs_secondary):
                    sec_local = torch.nonzero(needs_secondary, as_tuple=True)[0]
                    sec_batch = idx[sec_local]

                    sec_x_global = x_global_group[sec_local]
                    sec_mask = mask_batch.selection_masks[htp][sec_local]

                    sec_entities = (
                        entity_tensors[htp][sec_batch]
                        if entity_tensors[htp] is not None
                        else None
                    )

                    sel_head = self._selection_heads[htp]
                    if sec_entities is not None:
                        sec_out = sel_head(sec_entities, sec_x_global, sec_mask, sample)
                    else:
                        # HeadRelicSelect: x_global only
                        sec_out = sel_head(None, sec_x_global, sec_mask, sample)

                    if sample:
                        selection_indices[sec_batch] = sec_out.indices
                        selection_log_probs[sec_batch] = sec_out.log_probs
                    else:
                        selection_indices[sec_batch] = torch.argmax(sec_out.logits, dim=-1)

                    # ---- Inline target dispatch: COMBAT_DEFAULT play-card with target ----
                    if htp == _HTP_COMBAT_DEFAULT:
                        chosen_idx_hand = (
                            sec_out.indices
                            if sample
                            else torch.argmax(sec_out.logits, dim=-1)
                        )
                        # Look up target_required for the chosen card per sample
                        req = mask_batch.target_required[sec_batch, chosen_idx_hand]
                        if torch.any(req):
                            tgt_local = torch.nonzero(req, as_tuple=True)[0]
                            tgt_batch = sec_batch[tgt_local]
                            tgt_idx_hand = chosen_idx_hand[tgt_local]

                            # Card embedding for the chosen hand index
                            x_active_card = core_out.x_hand[tgt_batch, tgt_idx_hand]
                            tgt_x_global = x_global_group[sec_local][tgt_local]
                            tgt_monsters = core_out.x_monsters[tgt_batch]
                            tgt_mask = mask_batch.monster_alive_mask[tgt_batch]

                            tgt_out = self.head_monster_select(
                                tgt_monsters, tgt_x_global, tgt_mask, sample,
                                x_active_card=x_active_card,
                            )

                            if sample:
                                target_indices[tgt_batch] = tgt_out.indices
                                target_log_probs[tgt_batch] = tgt_out.log_probs
                            else:
                                target_indices[tgt_batch] = torch.argmax(tgt_out.logits, dim=-1)

            else:
                # ---- Direct primary: selection head only ----
                entities_group = entity_tensors[htp][idx]
                sel_mask = mask_batch.selection_masks[htp]
                sel_head = self._selection_heads[htp]
                sel_out = sel_head(entities_group, x_global_group, sel_mask, sample)

                if sample:
                    selection_indices[idx] = sel_out.indices
                    selection_log_probs[idx] = sel_out.log_probs
                else:
                    selection_indices[idx] = torch.argmax(sel_out.logits, dim=-1)

        return ForwardOutput(
            head_type_primaries=head_type_primaries,
            primary_indices=primary_indices,
            primary_log_probs=primary_log_probs,
            selection_indices=selection_indices,
            selection_log_probs=selection_log_probs,
            target_indices=target_indices,
            target_log_probs=target_log_probs,
            retain_indices=retain_indices,
            retain_log_probs=retain_log_probs,
            values=values,
        )

    def forward_single(
        self,
        x_game_state: XGameState,
        mask_batch: MaskBatch,
        sample: bool = True,
    ) -> SingleOutput:
        """Convenience method for single-sample inference."""
        out = self.forward(x_game_state, mask_batch, sample)
        ri_list = [int(x) for x in out.retain_indices[0].tolist() if x >= 0]
        return SingleOutput(
            head_type_primary=HeadTypePrimary(out.head_type_primaries[0].item()),
            primary_index=out.primary_indices[0].item(),
            primary_log_prob=out.primary_log_probs[0],
            selection_index=out.selection_indices[0].item(),
            selection_log_prob=out.selection_log_probs[0],
            target_index=out.target_indices[0].item(),
            target_log_prob=out.target_log_probs[0],
            retain_indices=ri_list,
            retain_log_prob=out.retain_log_probs[0],
            value=out.values[0],
        )
