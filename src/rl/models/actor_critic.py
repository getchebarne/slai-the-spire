"""Actor-Critic with screen/pending-routed hierarchical action heads.

Each sample is routed (by route.py) to a primary head:
  - Screen heads (COMBAT/MAP/REST/REWARD/SHOP/EVENT/CHEST): an option-kind
    categorical (HeadBinaryChoice with num_choices = #kinds), then for the chosen
    kind a secondary entity selection (per SelKey), and for CARD_PLAY / USE_POTION
    a tertiary inline monster target.
  - Pending multi-pick (PEND_DISCARD/RETAIN): HeadCardMultiPick over hand.
  - Pending single-select (setup/nightmare/discover/deck-pick): HeadEntitySelection.

Selection heads over the engine's per-pile transformer embeddings (hand/deck/
reward) consume CoreOutput; heads over screen-specific pools (potions/shop/event/
discover) consume the raw TensorGameState encodings directly with their own input
projection. The total action log-prob is option + selection + target + retain.
"""

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn

from src.rl.action_space.masks import SEL_POOL_SIZE
from src.rl.action_space.masks import MaskBatch
from src.rl.action_space.types import MULTIPICK_HEADS
from src.rl.action_space.types import NUM_SEL_KEYS
from src.rl.action_space.types import PEND_SINGLE_SELKEY
from src.rl.action_space.types import PRIMARY_NUM_CHOICES
from src.rl.action_space.types import SCREEN_HEADS
from src.rl.action_space.types import SCREEN_OPTION_KINDS
from src.rl.action_space.types import HeadTypePrimary
from src.rl.action_space.types import OptKind
from src.rl.action_space.types import SelKey
from src.rl.action_space.types import opt_may_target
from src.rl.action_space.types import opt_sel_key
from src.rl.action_space.types import to_action
from src.rl.constants import MAP_WIDTH
from src.rl.constants import MAX_SIZE_HAND
from src.rl.encoding.card import get_encoding_dim_card
from src.rl.encoding.event import get_encoding_dim_event_option
from src.rl.encoding.potion import get_encoding_dim_potion
from src.rl.encoding.shop import get_encoding_dim_shop_card
from src.rl.encoding.shop import get_encoding_dim_shop_potion
from src.rl.encoding.shop import get_encoding_dim_shop_relic
from src.rl.encoding.state import TensorGameState
from src.rl.models.core import Core
from src.rl.models.core import CoreOutput
from src.rl.models.heads import HeadBinaryChoice
from src.rl.models.heads import HeadCardMultiPick
from src.rl.models.heads import HeadEntitySelection
from src.rl.models.heads import HeadMapSelect
from src.rl.models.heads import HeadMonsterSelect
from src.rl.models.heads import HeadValue
from src.rl.models.heads import recompute_grouped_log_prob_and_entropy


# Per-SelKey pool source: ("core"|"xgs", attr) or ("map", None). Input dim filled
# at model init (dim_entity for core piles; raw encoding dim for xgs pools).
_SEL_SOURCE: dict = {
    SelKey.CARD_PLAY: ("core", "x_hand"),
    SelKey.POTION_USE: ("xgs", "x_potions"),
    SelKey.POTION_DISCARD: ("xgs", "x_potions"),
    SelKey.ROOM_SELECT: ("map", None),
    SelKey.REST_UPGRADE: ("core", "x_deck"),
    SelKey.REWARD_CARD: ("core", "x_combat_reward"),
    SelKey.SHOP_CARD: ("xgs", "x_shop_cards"),
    SelKey.SHOP_RELIC: ("xgs", "x_shop_relics"),
    SelKey.SHOP_POTION: ("xgs", "x_shop_potions"),
    SelKey.SHOP_PURGE: ("core", "x_deck"),
    SelKey.EVENT_OPTION: ("xgs", "x_event_options"),
    SelKey.PEND_SETUP: ("core", "x_hand"),
    SelKey.PEND_NIGHTMARE: ("core", "x_hand"),
    SelKey.PEND_DISCOVER: ("xgs", "x_discover"),
    SelKey.PEND_PURGE: ("core", "x_deck"),
    SelKey.PEND_UPGRADE: ("core", "x_deck"),
    SelKey.PEND_DUPLICATE: ("core", "x_deck"),
    SelKey.PEND_TRANSFORM: ("core", "x_deck"),
}


def _raw_pool_dim(attr: str) -> int:
    return {
        "x_potions": get_encoding_dim_potion(),
        "x_shop_cards": get_encoding_dim_shop_card(),
        "x_shop_relics": get_encoding_dim_shop_relic(),
        "x_shop_potions": get_encoding_dim_shop_potion(),
        "x_event_options": get_encoding_dim_event_option(),
        "x_discover": get_encoding_dim_card(),
    }[attr]


class ForwardOutput(NamedTuple):
    head_type_primaries: torch.Tensor  # (B,)
    option_indices: torch.Tensor  # (B,) -1 for pending heads
    option_log_probs: torch.Tensor  # (B,)
    selection_indices: torch.Tensor  # (B,) -1 if none
    selection_log_probs: torch.Tensor  # (B,)
    sel_keys: torch.Tensor  # (B,) recorded SelKey int, -1 if none
    target_indices: torch.Tensor  # (B,) -1 if untargeted
    target_log_probs: torch.Tensor  # (B,)
    retain_indices: torch.Tensor  # (B, MAX_SIZE_HAND) -1 in unused
    retain_log_probs: torch.Tensor  # (B,)
    values: torch.Tensor  # (B, 1)

    def get_action(self, idx: int):
        htp = HeadTypePrimary(int(self.head_type_primaries[idx].item()))
        ri = [int(x) for x in self.retain_indices[idx].tolist() if x >= 0]
        return to_action(
            htp,
            int(self.option_indices[idx].item()),
            int(self.selection_indices[idx].item()),
            target_index=int(self.target_indices[idx].item()),
            retain_indices=ri,
        )

    def get_log_prob(self, idx: int) -> torch.Tensor:
        return (
            self.option_log_probs[idx]
            + self.selection_log_probs[idx]
            + self.target_log_probs[idx]
            + self.retain_log_probs[idx]
        )


@dataclass
class SingleOutput:
    head_type_primary: HeadTypePrimary
    option_index: int
    selection_index: int
    target_index: int
    retain_indices: list
    value: torch.Tensor

    def to_action(self):
        return to_action(
            self.head_type_primary,
            self.option_index,
            self.selection_index,
            target_index=self.target_index,
            retain_indices=self.retain_indices,
        )


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
        self._dim_entity = dim_entity

        # Option-kind categorical heads, one per screen head (keyed by str(htp))
        self.option_heads = nn.ModuleDict(
            {
                str(int(htp)): HeadBinaryChoice(
                    dim_global, dim_ff_primary, num_choices=PRIMARY_NUM_CHOICES[htp]
                )
                for htp in SCREEN_HEADS
            }
        )

        # Selection heads, one per SelKey (keyed by str(selkey))
        self.sel_heads = nn.ModuleDict()
        for selkey in SelKey:
            src, attr = _SEL_SOURCE[selkey]
            if src == "map":
                self.sel_heads[str(int(selkey))] = HeadMapSelect(
                    dim_map, dim_global, dim_ff_map, MAP_WIDTH
                )
            elif src == "core":
                self.sel_heads[str(int(selkey))] = HeadEntitySelection(
                    dim_entity, dim_global, dim_ff_card
                )
            else:
                self.sel_heads[str(int(selkey))] = HeadEntitySelection(
                    _raw_pool_dim(attr), dim_global, dim_ff_card
                )

        # Pending multi-pick heads (separate params per the existing convention)
        self.multipick_heads = nn.ModuleDict(
            {
                str(int(htp)): HeadCardMultiPick(dim_entity, dim_global, dim_ff_card)
                for htp in MULTIPICK_HEADS
            }
        )

        # Inline monster target (shared by CARD_PLAY + USE_POTION)
        self.head_monster_select = HeadMonsterSelect(dim_entity, dim_global, dim_ff_monster)

        # Critic
        self.head_value = HeadValue(dim_global, dim_ff_value)

    # ---- pool helpers ----

    def _pool(self, selkey: int, core_out: CoreOutput, xgs: TensorGameState) -> torch.Tensor:
        src, attr = _SEL_SOURCE[selkey]
        if src == "core":
            return getattr(core_out, attr)
        if src == "xgs":
            return getattr(xgs, attr)
        return core_out.x_map  # map (unused as a sequence)

    def _run_selection(
        self,
        selkey: int,
        sub: torch.Tensor,
        core_out: CoreOutput,
        xgs: TensorGameState,
        mask_batch: MaskBatch,
        sample: bool,
    ):
        """Run the SelKey's head on sub-batch `sub`. Returns (indices, log_probs).
        log_probs is None when sample=False."""
        xg = core_out.x_global[sub]
        mask = mask_batch.sel_masks[selkey][sub]
        gids_full = mask_batch.sel_group_ids[selkey]
        gids = gids_full[sub] if gids_full is not None else None
        head = self.sel_heads[str(int(selkey))]

        src, _ = _SEL_SOURCE[selkey]
        if src == "map":
            out = head(core_out.x_map[sub], xg, mask, sample)
        else:
            pool = self._pool(selkey, core_out, xgs)[sub]
            out = head(pool, xg, mask, sample, group_ids=gids)

        if sample:
            return out.indices, out.log_probs
        return torch.argmax(out.logits, dim=-1), None

    def forward(
        self, x_game_state: TensorGameState, mask_batch: MaskBatch, sample: bool = True
    ) -> ForwardOutput:
        device = x_game_state.x_hand.device
        core_out = self.core(x_game_state)
        B = core_out.x_global.shape[0]

        values = self.head_value(core_out.x_global)

        htps = torch.full((B,), -1, dtype=torch.long, device=device)
        opt_idx = torch.full((B,), -1, dtype=torch.long, device=device)
        opt_lp = torch.zeros(B, device=device)
        sel_idx = torch.full((B,), -1, dtype=torch.long, device=device)
        sel_lp = torch.zeros(B, device=device)
        sel_keys = torch.full((B,), -1, dtype=torch.long, device=device)
        tgt_idx = torch.full((B,), -1, dtype=torch.long, device=device)
        tgt_lp = torch.zeros(B, device=device)
        retain_idx = torch.full((B, MAX_SIZE_HAND), -1, dtype=torch.long, device=device)
        retain_lp = torch.zeros(B, device=device)

        # ---- Screen heads ----
        for htp in SCREEN_HEADS:
            idx = mask_batch.route[htp]
            if len(idx) == 0:
                continue
            htps[idx] = int(htp)
            opt_head = self.option_heads[str(int(htp))]
            out = opt_head(core_out.x_global[idx], mask_batch.option_masks[htp], sample)
            if sample:
                chosen = out.indices
                opt_lp[idx] = out.log_probs
            else:
                chosen = torch.argmax(out.logits, dim=-1)
            opt_idx[idx] = chosen

            opts = SCREEN_OPTION_KINDS[htp]
            for pos, opt in enumerate(opts):
                selkey = opt_sel_key(opt)
                if selkey is None:
                    continue
                local = torch.nonzero(chosen == pos, as_tuple=True)[0]
                if local.numel() == 0:
                    continue
                sub = idx[local]
                s_i, s_lp = self._run_selection(
                    int(selkey), sub, core_out, x_game_state, mask_batch, sample
                )
                sel_idx[sub] = s_i
                sel_keys[sub] = int(selkey)
                if sample:
                    sel_lp[sub] = s_lp

                if opt_may_target(opt):
                    self._maybe_target(
                        opt, sub, s_i, core_out, mask_batch, sample, tgt_idx, tgt_lp
                    )

        # ---- Pending multi-pick ----
        for htp in MULTIPICK_HEADS:
            idx = mask_batch.route[htp]
            if len(idx) == 0:
                continue
            htps[idx] = int(htp)
            head = self.multipick_heads[str(int(htp))]
            mp = head(
                core_out.x_hand[idx],
                core_out.x_global[idx],
                mask_batch.multipick_mask[idx],
                mask_batch.pick_nums[idx],
                sample=sample,
                group_ids=mask_batch.multipick_group_ids[idx],
            )
            retain_idx[idx] = mp.indices
            if sample:
                retain_lp[idx] = mp.log_prob

        # ---- Pending single-select ----
        for htp, selkey in PEND_SINGLE_SELKEY.items():
            idx = mask_batch.route[htp]
            if len(idx) == 0:
                continue
            htps[idx] = int(htp)
            s_i, s_lp = self._run_selection(
                int(selkey), idx, core_out, x_game_state, mask_batch, sample
            )
            sel_idx[idx] = s_i
            sel_keys[idx] = int(selkey)
            if sample:
                sel_lp[idx] = s_lp

        return ForwardOutput(
            head_type_primaries=htps,
            option_indices=opt_idx,
            option_log_probs=opt_lp,
            selection_indices=sel_idx,
            selection_log_probs=sel_lp,
            sel_keys=sel_keys,
            target_indices=tgt_idx,
            target_log_probs=tgt_lp,
            retain_indices=retain_idx,
            retain_log_probs=retain_lp,
            values=values,
        )

    def _maybe_target(self, opt, sub, s_i, core_out, mask_batch, sample, tgt_idx, tgt_lp) -> None:
        """Inline monster target for CARD_PLAY / USE_POTION when the chosen entity
        requires a target. CARD_PLAY passes the played card embedding as the
        active context; USE_POTION passes none (zeros)."""
        if opt == OptKind.CARD_PLAY:
            req = mask_batch.target_required_hand[sub, s_i]
        else:
            req = mask_batch.target_required_potion[sub, s_i]
        tlocal = torch.nonzero(req, as_tuple=True)[0]
        if tlocal.numel() == 0:
            return
        tsub = sub[tlocal]
        x_active = None
        if opt == OptKind.CARD_PLAY:
            x_active = core_out.x_hand[tsub, s_i[tlocal]]
        out = self.head_monster_select(
            core_out.x_monsters[tsub],
            core_out.x_global[tsub],
            mask_batch.monster_alive_mask[tsub],
            sample,
            x_active_card=x_active,
        )
        if sample:
            tgt_idx[tsub] = out.indices
            tgt_lp[tsub] = out.log_probs
        else:
            tgt_idx[tsub] = torch.argmax(out.logits, dim=-1)

    # ---- PPO recompute ----

    def _eval_selection(self, selkey, sub, rec_idx_sub, core_out, xgs, mask_batch):
        xg = core_out.x_global[sub]
        mask = mask_batch.sel_masks[selkey][sub]
        gids_full = mask_batch.sel_group_ids[selkey]
        gids = gids_full[sub] if gids_full is not None else None
        head = self.sel_heads[str(int(selkey))]
        src, _ = _SEL_SOURCE[selkey]
        if src == "map":
            out = head(core_out.x_map[sub], xg, mask, sample=False)
        else:
            pool = self._pool(selkey, core_out, xgs)[sub]
            out = head(pool, xg, mask, sample=False, group_ids=gids)
        return recompute_grouped_log_prob_and_entropy(out.logits, rec_idx_sub, group_ids=gids)

    def evaluate_actions(self, x_game_state, mask_batch, rec):
        """Recompute log-probs/entropies of recorded actions under the current
        policy, mirroring forward(). `rec` carries recorded
        head_type_primaries/option_indices/selection_indices/target_indices/
        retain_indices (all (B,) / (B,H)). Returns (log_probs, entropies, values)."""
        core_out = self.core(x_game_state)
        xg = core_out.x_global
        B = xg.shape[0]
        device = xg.device
        values = self.head_value(xg)
        log_probs = torch.zeros(B, device=device)
        entropies = torch.zeros(B, device=device)

        for htp in SCREEN_HEADS:
            idx = mask_batch.route[htp]
            if len(idx) == 0:
                continue
            logits = self.option_heads[str(int(htp))](
                xg[idx], mask_batch.option_masks[htp], sample=False
            ).logits
            dist = torch.distributions.Categorical(logits=logits)
            rec_opt = rec.option_indices[idx]
            log_probs[idx] += dist.log_prob(rec_opt)
            entropies[idx] += dist.entropy()

            opts = SCREEN_OPTION_KINDS[htp]
            for pos, opt in enumerate(opts):
                selkey = opt_sel_key(opt)
                if selkey is None:
                    continue
                local = torch.nonzero(rec_opt == pos, as_tuple=True)[0]
                if local.numel() == 0:
                    continue
                sub = idx[local]
                rec_sel = rec.selection_indices[sub]
                lp, e = self._eval_selection(
                    int(selkey), sub, rec_sel, core_out, x_game_state, mask_batch
                )
                log_probs[sub] += lp
                entropies[sub] += e

                if opt_may_target(opt):
                    rec_tgt = rec.target_indices[sub]
                    has = rec_tgt >= 0
                    tl = torch.nonzero(has, as_tuple=True)[0]
                    if tl.numel() == 0:
                        continue
                    tsub = sub[tl]
                    x_active = (
                        core_out.x_hand[tsub, rec_sel[tl]] if opt == OptKind.CARD_PLAY else None
                    )
                    out = self.head_monster_select(
                        core_out.x_monsters[tsub],
                        xg[tsub],
                        mask_batch.monster_alive_mask[tsub],
                        sample=False,
                        x_active_card=x_active,
                    )
                    lp, e = recompute_grouped_log_prob_and_entropy(out.logits, rec_tgt[tl])
                    log_probs[tsub] += lp
                    entropies[tsub] += e

        for htp in MULTIPICK_HEADS:
            idx = mask_batch.route[htp]
            if len(idx) == 0:
                continue
            lp, e = self.multipick_heads[str(int(htp))].recompute_log_prob(
                core_out.x_hand[idx],
                xg[idx],
                mask_batch.multipick_mask[idx],
                mask_batch.pick_nums[idx],
                rec.retain_indices[idx],
                group_ids=mask_batch.multipick_group_ids[idx],
            )
            log_probs[idx] += lp
            entropies[idx] += e

        for htp, selkey in PEND_SINGLE_SELKEY.items():
            idx = mask_batch.route[htp]
            if len(idx) == 0:
                continue
            rec_sel = rec.selection_indices[idx]
            lp, e = self._eval_selection(
                int(selkey), idx, rec_sel, core_out, x_game_state, mask_batch
            )
            log_probs[idx] += lp
            entropies[idx] += e

        return log_probs, entropies, values

    def forward_single(
        self, x_game_state: TensorGameState, mask_batch: MaskBatch, sample: bool = True
    ) -> SingleOutput:
        out = self.forward(x_game_state, mask_batch, sample)
        ri = [int(x) for x in out.retain_indices[0].tolist() if x >= 0]
        return SingleOutput(
            head_type_primary=HeadTypePrimary(int(out.head_type_primaries[0].item())),
            option_index=int(out.option_indices[0].item()),
            selection_index=int(out.selection_indices[0].item()),
            target_index=int(out.target_indices[0].item()),
            retain_indices=ri,
            value=out.values[0],
        )
