"""Actor-Critic with a mask-derived 3-level action hierarchy (no route module).

Routing + every mask come from the engine's legal actions (masks.py). Per (B,) batch:
  - L1 option: ONE masked categorical over slai.ActionType (which action KIND), run for
    every row. A halt is just a one-legal-kind state, so its forced pick contributes
    log-prob 0 — no pending special case.
  - L2 selection: pointer network (AlphaStar-style) — entity-class key projections
    (shared across the pools of a class) dotted with a query from ONE shared query net
    over [x_global, ActionType embedding]; shop pools add a price-key term.
  - L3 target: the same pointer geometry over monsters, query conditioned on the
    active entity (played card / thrown potion), masked by the per-entity legal
    target set the engine enumerates (one CardPlay/PotionUse per monster).

The total action log-prob is option + selection + target.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.rl.index import GLOBAL_SLICE
from src.rl.index import Token
from src.rl.index import TokenContext
from src.rl.index import TokenKind
from src.rl.types import TMask
from src.rl.types import AT_MAY_TARGET
from src.rl.types import AT_TARGET
from src.rl.types import MAP_TARGET
from src.rl.types import NUM_ACTION_TYPES
from src.rl.types import action_from_actiontype
from src.rl.encoding.map_ import MAP_NUM_ROOM_KINDS
from src.rl.encoding.screen import _ENCODING_DIM_SCREEN
from src.rl.encoding.shop import ENCODING_DIM_PRICE
from src.rl.types import TGameState
from src.rl.models.core import Core
from src.rl.models.heads import HeadBinaryChoice
from src.rl.models.heads import HeadPointerSelect
from src.rl.models.heads import HeadValue
from src.rl.models.heads import PointerKeys
from src.rl.reward import REWARD_STREAMS


# Shop price tensor per token kind: a SHOP-context selection adds an additive price-key
# term over the matching per-item price tensor on TCoreOutput.
_SHOP_PRICE_FIELD: dict[TokenKind, str] = {
    TokenKind.CARD: "shop_card_prices",
    TokenKind.RELIC: "shop_relic_prices",
    TokenKind.POTION: "shop_potion_prices",
}


@dataclass
class Pick:
    """One sampled head-pick (option | selection | target) over the (B,) batch: the
    index a masked Categorical chose, plus its log-prob and entropy. idx is -1 where
    the pick doesn't apply to a row; log_prob/entropy are 0 there (sum-neutral).
    `write` scatters an `_categorical_step` result, skipping None parts, so one call site serves
    sampling (idx, lp), greedy (idx) and recompute (lp, ent)."""

    idx: torch.Tensor  # (B,) long, -1 where N/A
    log_prob: torch.Tensor  # (B,) float, 0 where N/A
    entropy: torch.Tensor  # (B,) float, 0 where N/A

    @classmethod
    def na(cls, B: int, device: torch.device) -> "Pick":
        return cls(
            torch.full((B,), -1, dtype=torch.long, device=device),
            torch.zeros(B, device=device),
            torch.zeros(B, device=device),
        )

    def write(self, rows, idx, log_prob, entropy) -> None:
        if idx is not None:
            self.idx[rows] = idx
        if log_prob is not None:
            self.log_prob[rows] = log_prob
        if entropy is not None:
            self.entropy[rows] = entropy


@dataclass
class ActionBatch:
    """One routing pass over a (B,) batch — the forward() return and the _run result.
    Mirrors the action's structure: three sampled picks (option -> selection -> target),
    each an (idx, log_prob, entropy) triple. `option.idx` is the chosen ActionType (L1
    always runs, masked to the legal kinds), which alone determines emission. The action's
    log-prob/entropy are the sums of the picks'. Shared by sampling (forward) and PPO
    recompute (evaluate_actions)."""

    values: torch.Tensor  # (B, len(REWARD_STREAMS)) critic, one output per reward stream
    option: Pick  # L1: chosen ActionType (idx), its log-prob, entropy
    selection: Pick
    target: Pick

    @classmethod
    def empty(cls, B: int, device: torch.device, values: torch.Tensor) -> "ActionBatch":
        return cls(
            values=values,
            option=Pick.na(B, device),
            selection=Pick.na(B, device),
            target=Pick.na(B, device),
        )

    def total_log_prob(self) -> torch.Tensor:
        return self.option.log_prob + self.selection.log_prob + self.target.log_prob

    def get_action(self, i: int):
        return action_from_actiontype(
            int(self.option.idx[i].item()),  # the chosen ActionType
            int(self.selection.idx[i].item()),
            int(self.target.idx[i].item()),
        )

    def get_log_prob(self, i: int) -> torch.Tensor:
        return self.option.log_prob[i] + self.selection.log_prob[i] + self.target.log_prob[i]


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
        dim_ff_value: int = 128,
        dim_op: int = 32,
        dim_key: int = 32,
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

        # L1: one masked categorical over ActionType (which action kind), screen-gated
        # (GLU). Pending-only types are always masked here; a halt has an empty L1 mask.
        self.option_head = HeadBinaryChoice(
            dim_global,
            dim_ff_primary,
            num_choices=NUM_ACTION_TYPES,
            dim_context=_ENCODING_DIM_SCREEN,
        )

        # L2/L3: pointer selection — one key projection per TokenKind (keyed by
        # kind name; MONSTER serves L3) + one shared query net per level. Every target
        # except MAP is a slice of the refined token tensor (index.GLOBAL_SLICE[token]);
        # SHOP-context targets add an additive price-key term over their price tensors.
        self.operation_embedding = nn.Embedding(NUM_ACTION_TYPES, dim_op)
        self.pointer_keys = nn.ModuleDict(
            {
                "CARD": PointerKeys(dim_entity, dim_key),
                "POTION": PointerKeys(dim_entity, dim_key),
                "RELIC": PointerKeys(dim_entity, dim_key),
                "EVENT": PointerKeys(dim_entity, dim_key),
                "MONSTER": PointerKeys(dim_entity, dim_key),
                "MAP": PointerKeys(dim_map + MAP_NUM_ROOM_KINDS, dim_key),
            }
        )
        self.price_keys = nn.Linear(ENCODING_DIM_PRICE, dim_key, bias=False)
        self.query_l2 = HeadPointerSelect(dim_global, dim_op, dim_key)
        self.query_l3 = HeadPointerSelect(dim_global, dim_entity, dim_key)

        # Critic: one output per reward stream (value decomposition)
        self.head_value = HeadValue(dim_global, dim_ff_value, len(REWARD_STREAMS))

        # PPO init: orthogonal everywhere (gain √2), then near-zero logit layers
        # (initial policy ≈ uniform over the mask) and a unit-gain value output.
        # For the pointer heads the logit-producing layer is the query output (small
        # query -> logits ≈ 0); keys keep gain √2.
        # Embeddings/LayerNorm/attention in_proj keep PyTorch defaults.
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:  # price_keys is bias-free
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.option_head._scorer[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.query_l2._query_net[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.query_l3._query_net[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.head_value._network[-1].weight, gain=1.0)

    # ---- One routing pass, shared by sampling and PPO recompute ----

    def _categorical_step(self, logits, mask, sample, rec_idx=None):
        """One masked-categorical step. Identity dedup is baked into `mask`, so this
        is a plain Categorical (no grouped sampling).

        rec_idx given -> recompute: (None, log_prob, entropy) of the recorded pick.
        else sample   -> (sampled indices, log_prob, None).
        else greedy   -> (argmax indices, None, None).
        """
        masked = logits.masked_fill(~mask, float("-inf"))
        dist = torch.distributions.Categorical(logits=masked)
        if rec_idx is not None:
            return None, dist.log_prob(rec_idx), dist.entropy()
        if sample:
            idx = dist.sample()
            return idx, dist.log_prob(idx), None
        return torch.argmax(masked, dim=-1), None, None

    def _selection_step(self, at, rows, core_out, mask_batch, sample, rec_idx=None):
        """Score + sample/recompute the selection for one action type `at` on `rows`.
        The mask is per action type (legality differs per type); keys come from the
        target kind's projection (shared across that kind's tokens), the query from the
        shared L2 query net conditioned on the ActionType embedding."""
        target = AT_TARGET[at]
        mask = mask_batch.mask_action_idx[str(at)][rows]
        if target is MAP_TARGET:
            # Concat per-column next-row room kinds onto the column embeddings so the
            # pointer keys (not just the shared query) can tell the columns apart.
            pool_tensor = torch.cat(
                [core_out.x_map[rows], core_out.x_map_candidates[rows]], dim=-1
            )
            keys = self.pointer_keys["MAP"](pool_tensor)
        else:
            pool_tensor = core_out.tokens.x[:, GLOBAL_SLICE[target]][rows]
            keys = self.pointer_keys[target.kind.name](pool_tensor)
            if target.context is TokenContext.SHOP:  # shop tokens: additive price-key term
                price = getattr(core_out, _SHOP_PRICE_FIELD[target.kind])
                keys = keys + self.price_keys(price[rows])
        x_op = self.operation_embedding(torch.full_like(rows, at))
        logits = self.query_l2(keys, core_out.x_global[rows], x_op, mask).logits
        return self._categorical_step(logits, mask, sample, rec_idx=rec_idx)

    def _target_step(self, target, rows, sel_idx, core_out, mask_batch, sample, rec_tgt=None):
        """Inline monster target for a CardPlay (CARD/HAND) / PotionUse (POTION) group. The
        legal-monster set is the engine-enumerated per-entity target mask; the gate is
        that set being non-empty (sampling) or the recorded target's presence (recompute).
        Returns None if no row needs a target, else (tsub, indices, log_prob, entropy)."""
        tmask = (
            mask_batch.mask_target_card
            if target.kind == TokenKind.CARD
            else mask_batch.mask_target_potion
        )
        monster_mask = tmask[rows, sel_idx]  # (len(rows), MAX_MONSTERS)
        need = rec_tgt >= 0 if rec_tgt is not None else monster_mask.any(dim=-1)
        tl = torch.nonzero(need, as_tuple=True)[0]
        if tl.numel() == 0:
            return None
        tsub = rows[tl]
        mask = monster_mask[tl]
        x_active = core_out.tokens.x[:, GLOBAL_SLICE[target]][tsub, sel_idx[tl]]
        keys = self.pointer_keys["MONSTER"](
            core_out.tokens.x[:, GLOBAL_SLICE[Token(TokenKind.MONSTER, TokenContext.ENEMIES)]][
                tsub
            ]
        )
        logits = self.query_l3(keys, core_out.x_global[tsub], x_active, mask).logits
        rec_idx = rec_tgt[tl] if rec_tgt is not None else None
        idx, lp, ent = self._categorical_step(logits, mask, sample, rec_idx=rec_idx)
        return tsub, idx, lp, ent

    def _run(
        self,
        core_out,
        mask_batch,
        sample,
        values,
        rec_option=None,
        rec_selection=None,
        rec_target=None,
    ):
        """Single mask-derived routing pass. rec_* = None -> sample/greedy; rec_* given ->
        recompute the recorded action. L1 (over mask_action_type) runs for every row — a
        halt is just a one-legal-kind state whose forced pick contributes log-prob 0. The
        chosen ActionType selects the L2 group; L3 targets the may-target groups.
        `_categorical_step` returns idx=None on recompute, so passing its result into
        Pick.write skips the recorded index."""
        recompute = rec_option is not None
        B = core_out.x_global.shape[0]
        device = core_out.x_global.device
        s = ActionBatch.empty(B, device, values)

        # ---- L1: which action kind (always; masked to the legal kinds) ----
        logits = self.option_head(
            core_out.x_global, core_out.x_screen, mask_batch.mask_action_type
        ).logits
        chosen, lp, ent = self._categorical_step(
            logits, mask_batch.mask_action_type, sample, rec_idx=rec_option
        )
        s.option.write(torch.arange(B, device=device), chosen, lp, ent)
        decided = rec_option if recompute else chosen  # (B,) chosen ActionType per row

        # ---- L2: selection, grouped by ActionType (mask per type, head per pool) ----
        # One stable argsort groups the rows by decided type (contiguous slices),
        # replacing a nonzero scan per action type. Terminal types appear in
        # `decided` too; iterating the L2 mask keys filters them out as before.
        order = torch.argsort(decided, stable=True)
        type_counts = torch.bincount(decided, minlength=NUM_ACTION_TYPES).tolist()
        offsets = [0]
        for count in type_counts:
            offsets.append(offsets[-1] + count)
        for key in mask_batch.mask_action_idx.keys():
            at = int(key)
            if type_counts[at] == 0:
                continue
            rows = order[offsets[at] : offsets[at + 1]]
            rec_sel = rec_selection[rows] if recompute else None
            s_i, s_lp, s_ent = self._selection_step(
                at, rows, core_out, mask_batch, sample, rec_idx=rec_sel
            )
            s.selection.write(rows, s_i, s_lp, s_ent)

            # ---- L3: monster target (CardPlay / PotionUse groups) ----
            if AT_MAY_TARGET[at]:
                sel_for_target = rec_sel if recompute else s_i
                rec_tgt = rec_target[rows] if recompute else None
                res = self._target_step(
                    AT_TARGET[at],
                    rows,
                    sel_for_target,
                    core_out,
                    mask_batch,
                    sample,
                    rec_tgt=rec_tgt,
                )
                if res is not None:
                    tsub, t_i, t_lp, t_ent = res
                    s.target.write(tsub, t_i, t_lp, t_ent)

        return s

    def forward(
        self, x_game_state: TGameState, mask_batch: TMask, sample: bool = True
    ) -> ActionBatch:
        core_out = self.core(x_game_state)
        values = self.head_value(core_out.x_global)
        return self._run(core_out, mask_batch, sample, values)

    def evaluate_actions(
        self, x_game_state, mask_batch, option_indices, selection_indices, target_indices
    ):
        """Recompute log-probs/entropies of the recorded (option, selection, target)
        indices under the current policy, sharing forward()'s routing pass.
        Returns (log_probs, entropies, values); entropies is a per-head (B, 3) stack
        [option, selection, target] (0 where the pick doesn't apply), so the loss
        sums it and logging can split it; values is (B, len(REWARD_STREAMS))."""
        core_out = self.core(x_game_state)
        values = self.head_value(core_out.x_global)
        s = self._run(
            core_out,
            mask_batch,
            sample=False,
            values=values,
            rec_option=option_indices,
            rec_selection=selection_indices,
            rec_target=target_indices,
        )
        entropies = torch.stack([s.option.entropy, s.selection.entropy, s.target.entropy], dim=-1)
        return s.total_log_prob(), entropies, values
