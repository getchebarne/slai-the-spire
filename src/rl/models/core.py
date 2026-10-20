from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.encoding.energy import ENCODING_DIM_ENERGY
from src.rl.encoding.event import ENCODING_DIM_EVENT_META
from src.rl.encoding.focus import ENCODING_DIM_FOCUS
from src.rl.encoding.focus import OFF_FOCUS_DRAW_ORDER_VISIBLE
from src.rl.encoding.map_ import ENCODING_DIM_MAP_META
from src.rl.encoding.reward import ENCODING_DIM_REWARD_META
from src.rl.encoding.shop import ENCODING_DIM_SHOP_META
from src.rl.encoding.state import ENCODING_DIM_ASCENSION
from src.rl.models.entity_projector import EntityProjector
from src.rl.models.entity_transformer import DrawOrderLayer
from src.rl.models.entity_transformer import EntityTransformer
from src.rl.models.map_encoder import MapGNN
from src.rl.types import CAP_OF_SLICE_KIND
from src.rl.types import SLICE_KINDS_CARD_REWARDS
from src.rl.types import SliceKind
from src.rl.types import X_CoreOutput
from src.rl.types import X_Entity
from src.rl.types import X_GameState


# Count features: one per slice kind except the character singleton (a constant 1.0), each
# normalized by its slice kind's cap (a count never exceeds it), in window order
_COUNT_SLICE_KINDS: tuple[SliceKind, ...] = (  # Explicit: checkpoints pin this column order
    SliceKind.CARD_PILE_HAND,
    SliceKind.CARD_PILE_DISCOVER,
    SliceKind.CARD_PILE_SELECTED,
    SliceKind.CARD_REWARD_0,
    SliceKind.CARD_REWARD_1,
    SliceKind.CARD_REWARD_2,
    SliceKind.CARD_REWARD_3,
    SliceKind.CARD_REWARD_4,
    SliceKind.CARD_SHOP,
    SliceKind.CARD_PILE_EVENT_ROLL,
    SliceKind.CARD_MATCH_FLIPPED,
    SliceKind.CARD_PILE_STASIS,
    SliceKind.CARD_PLAY_QUEUE,
    SliceKind.CARD_PILE_NIGHTMARE,
    SliceKind.CARD_PILE_DRAW,
    SliceKind.CARD_PILE_DISCARD,
    SliceKind.CARD_PILE_EXHAUST,
    SliceKind.CARD_PILE_DECK,
    SliceKind.RELIC_REWARD,
    SliceKind.RELIC_SHOP,
    SliceKind.RELIC_EVENT_ROLL,
    SliceKind.RELIC_OWNED,
    SliceKind.POTION_OWNED,
    SliceKind.POTION_REWARD,
    SliceKind.POTION_SHOP,
    SliceKind.POTION_EVENT_ROLL,
    SliceKind.MONSTERS,
    SliceKind.EVENT_OPTIONS,
    SliceKind.ROOMS,
)
_REWARD_LO = SLICE_KINDS_CARD_REWARDS[0]
_REWARD_HI = SLICE_KINDS_CARD_REWARDS[-1]

# The copies of a table that _SpreadEmbedding's backward sums over
_SPREAD_COPIES = 256


def draw_depth(x_slice_kind: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per window token, its depth in the draw pile: the cards above it, 0 off the
    draw tokens. Per row, whether the pile shows an order: at least 2 draw tokens.

    The draw tokens are the pile's top cards in engine order, bottom first, so the
    last rank is the next card drawn: depth = count - 1 - rank."""
    x_is_draw = x_slice_kind == SliceKind.CARD_PILE_DRAW
    x_count = x_is_draw.sum(dim=1, keepdim=True)

    # A draw token's running count is its rank + 1
    x_depth = (x_count - x_is_draw.cumsum(dim=1)) * x_is_draw
    return x_depth, x_count.squeeze(1) >= 2


class _SpreadEmbedding(torch.autograd.Function):
    """F.embedding, whose backward sums each table row's gradient over _SPREAD_COPIES copies of
    the table: the token at flat position i adds into copy i % _SPREAD_COPIES, then the copies
    add up. The same sums in another order, in short loops"""

    @staticmethod
    def forward(
        ctx: Any, x_weight: torch.Tensor, x_idxs: torch.Tensor, padding_idx: int | None
    ) -> torch.Tensor:
        ctx.save_for_backward(x_idxs)
        ctx.num_embeddings, ctx.padding_idx = x_weight.shape[0], padding_idx
        return F.embedding(x_idxs, x_weight)

    @staticmethod
    def backward(ctx: Any, x_grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (x_idxs,) = ctx.saved_tensors
        x_copy = (
            torch.arange(x_idxs.numel(), device=x_idxs.device).view_as(x_idxs) % _SPREAD_COPIES
        )
        x_grad_copies = torch.ops.aten.embedding_dense_backward(
            x_grad,
            x_idxs + x_copy * ctx.num_embeddings,
            ctx.num_embeddings * _SPREAD_COPIES,
            padding_idx=-1,
            scale_grad_by_freq=False,
        )
        x_grad_weight = x_grad_copies.view(_SPREAD_COPIES, ctx.num_embeddings, -1).sum(0)

        # As nn.Embedding's: the padding row gets no gradient
        if ctx.padding_idx is not None:
            x_grad_weight[ctx.padding_idx] = 0
        return x_grad_weight, None, None


def _embedding(table: nn.Embedding, x_idxs: torch.Tensor) -> torch.Tensor:
    """`table`'s rows at the (B, S) `x_idxs`, the gradient's sums spread by _SpreadEmbedding"""
    return _SpreadEmbedding.apply(table.weight, x_idxs, table.padding_idx)


class Core(nn.Module):
    def __init__(
        self,
        dim_entity: int,
        dim_global: int,
        transformer_dim_ff: int,
        transformer_num_heads: int,
        transformer_num_blocks: int,
        gnn_num_layers: int,
        map_encoder_dim: int,
    ) -> None:
        super().__init__()

        self._entity_projector = EntityProjector(dim_entity, map_encoder_dim)
        self._entity_transformer = EntityTransformer(
            dim_entity, transformer_dim_ff, transformer_num_heads, transformer_num_blocks
        )

        # Draw order: the value head reads the layer on every ordered pile, the policy only under
        # Frozen Eye
        self._draw_order_layer = DrawOrderLayer(
            dim_entity, transformer_dim_ff, transformer_num_heads
        )

        # Token identity: one row per SliceKind, looked up by the window's slice kind row
        self._type_emb = nn.Embedding(len(SliceKind), dim_entity)
        self._map_gnn = MapGNN(gnn_num_layers, map_encoder_dim)

        # Global context: global token + character + map readout + slice-kind counts + raw flats
        global_input_dim = (
            dim_entity  # global token (attention-aggregated entities)
            + dim_entity  # character (refined singleton)
            + map_encoder_dim  # map GNN readout (whole-graph summary)
            + len(_COUNT_SLICE_KINDS)  # per-slice-kind token counts, sqrt-normalized by their caps
            + ENCODING_DIM_ENERGY  # energy (raw)
            + ENCODING_DIM_FOCUS  # focus row
            + ENCODING_DIM_ASCENSION  # ascension thermometer (curriculum level)
            + ENCODING_DIM_MAP_META  # floor, act, boss, room kinds, "?"-room odds and event pools
            + ENCODING_DIM_REWARD_META
            + ENCODING_DIM_SHOP_META  # shop purge price triple + purged flag
            + ENCODING_DIM_EVENT_META
            + dim_entity  # embedded pending-halt effect (EntityProjector)
        )

        # Output LN: HeadActionType/HeadValue read this vector without their
        # own norm, so scale drift here would move logits directly
        self._global_projection = nn.Sequential(
            nn.Linear(global_input_dim, dim_global),
            nn.ReLU(),
            nn.Linear(dim_global, dim_global),
            nn.LayerNorm(dim_global),
        )

        # Count features: slice kinds to count and each slice kind's cap; derived, so excluded
        # from checkpoints
        self.register_buffer(
            "_count_slice_kinds", torch.tensor(_COUNT_SLICE_KINDS).view(1, 1, -1), persistent=False
        )
        self.register_buffer(
            "_count_norm",
            torch.tensor(
                [float(CAP_OF_SLICE_KIND[slice_kind]) for slice_kind in _COUNT_SLICE_KINDS]
            ).view(1, -1),
            persistent=False,
        )

        # The per-bundle slice kinds carry their own type rows; this shared embedding
        # is the global CARD_REWARD identity summed on top of all of them
        self._reward_emb = nn.Embedding(2, dim_entity, padding_idx=0)

    def forward(self, x_game_state: X_GameState, with_value: bool = True) -> X_CoreOutput:
        """The core's output; `with_value=False` runs the draw-order layer only on the rows the
        policy reads, for callers that need no value, and leaves `global_value` the policy's"""

        # Map GNN: next-row room features + whole-graph readout
        x_rooms, x_map_readout = self._map_gnn(
            x_game_state.map_grid, x_game_state.rooms.x_node_idx, x_game_state.rooms.x_slice_kind
        )

        # The window as one X_Entity (the six blocks then the rooms), its masks, its valid tokens
        # per entity type, and the embedded pending halt
        x_entities, x_legal, x_needs_target, num_tokens, x_pending = self._entity_projector(
            x_game_state, x_rooms
        )

        # Per-state inputs to the global context: the same for the transformer and the layer, so
        # cat'd once
        x_state_inputs = torch.cat(
            [
                x_map_readout,
                self._get_token_counts(x_entities.x_slice_kind),
                x_game_state.energy,
                x_game_state.focus,
                x_game_state.ascension,
                x_game_state.map_meta,
                x_game_state.reward_meta,
                x_game_state.shop_meta,
                x_game_state.event_meta,
                x_pending,
            ],
            dim=1,
        )

        # The layer's rows, whose pile shows an order: the Frozen Eye ones, then the rest unless
        # `with_value` is False. Their counts are read back with the transformer's packing widths
        x_depth, x_has_order = draw_depth(x_entities.x_slice_kind)
        x_draw_order_visible = x_game_state.focus[:, OFF_FOCUS_DRAW_ORDER_VISIBLE] > 0
        x_ordered_visible = x_has_order & x_draw_order_visible
        x_ordered_hidden = x_has_order & ~x_draw_order_visible

        # One transformer pass over every state, without the draw depths
        x_refined, x_global_token, (num_visible, num_hidden) = self._entity_transformer(
            self._embed_identity(x_entities),
            torch.stack([x_ordered_visible.sum(), x_ordered_hidden.sum()]),
        )
        x_global = self._global_context(x_refined, x_global_token, x_state_inputs)
        x_rows = torch.cat(
            [
                torch.nonzero_static(x_ordered_visible, size=num_visible),
                torch.nonzero_static(x_ordered_hidden, size=num_hidden if with_value else 0),
            ]
        ).squeeze(1)
        if x_rows.shape[0] == 0:
            return X_CoreOutput(
                global_=x_global,
                global_value=x_global,
                entities=x_refined,
                x_legal=x_legal,
                x_needs_target=x_needs_target,
                num_tokens=num_tokens,
            )

        # The value head reads the layer on every one of its rows
        x_refined_layer, x_global_token_layer = self._draw_order_layer(
            x_refined, x_global_token, x_depth, x_rows
        )
        x_global_layer = self._global_context(
            x_refined_layer, x_global_token_layer, x_state_inputs[x_rows]
        )
        x_global_value = x_global.index_copy(0, x_rows, x_global_layer)

        # The policy reads it on the Frozen Eye rows. Their window is a view of its flat tokens, as
        # the transformer's is: the compiled heads guard on the window's base
        if num_visible > 0:
            x_rows_visible = x_rows[:num_visible]
            num_slots = x_refined.x.shape[1]
            x_slots = x_rows_visible.unsqueeze(1) * num_slots + torch.arange(
                num_slots, device=x_rows.device
            )
            x_global = x_global.index_copy(0, x_rows_visible, x_global_layer[:num_visible])
            x_refined = X_Entity(
                x_refined.x.flatten(0, 1)
                .index_copy(0, x_slots.flatten(), x_refined_layer.x[:num_visible].flatten(0, 1))
                .view_as(x_refined.x),
                x_refined.x_slice_kind,
            )
        return X_CoreOutput(
            global_=x_global,
            global_value=x_global_value,
            entities=x_refined,
            x_legal=x_legal,
            x_needs_target=x_needs_target,
            num_tokens=num_tokens,
        )

    def _global_context(
        self, x_refined: X_Entity, x_global_token: torch.Tensor, x_state_inputs: torch.Tensor
    ) -> torch.Tensor:
        """The global vector of each state of a refined window, from its global token, its
        character token and its per-state inputs"""
        # The one character token per state, found by its slice kind: exactly one -> (B, D)
        x_is_character = x_refined.x_slice_kind == SliceKind.CHARACTER
        x_at = x_is_character.long().argmax(1)  # a boolean index would make the CPU wait
        x_character = x_refined.x[torch.arange(x_at.shape[0], device=x_at.device), x_at]
        return self._global_projection(
            torch.cat([x_global_token, x_character, x_state_inputs], dim=1)
        )

    def _embed_identity(self, x_entities: X_Entity) -> X_Entity:
        # Every token's identity comes from its slice kind: the slice kind's type row, plus the
        # shared CARD_REWARD identity on the reward slice kinds
        x_slice_kind = x_entities.x_slice_kind
        x_is_reward = (x_slice_kind >= _REWARD_LO) & (x_slice_kind <= _REWARD_HI)

        # Pads borrow type row 0; they are masked downstream
        x_type = _embedding(self._type_emb, x_slice_kind.clamp(min=0))
        return X_Entity(
            x_entities.x + x_type + _embedding(self._reward_emb, x_is_reward.long()), x_slice_kind
        )

    def _get_token_counts(self, x_slice_kind: torch.Tensor) -> torch.Tensor:
        """Per-slice-kind token counts, sqrt-normalized by the slice kind's cap (a count never
        exceeds it, so the ratio is in [0, 1])."""
        x_count = (x_slice_kind.unsqueeze(-1) == self._count_slice_kinds).sum(dim=1).float()
        return (x_count / self._count_norm).sqrt()


def _check_tables() -> None:
    """The count slice kinds cover every slice kind but the character, and the reward slice
    kinds are a range"""
    if set(_COUNT_SLICE_KINDS) != set(SliceKind) - {SliceKind.CHARACTER}:
        raise RuntimeError(
            "every slice kind counts once: _COUNT_SLICE_KINDS and every SliceKind but CHARACTER "
            f"differ on {set(_COUNT_SLICE_KINDS) ^ (set(SliceKind) - {SliceKind.CHARACTER})}"
        )
    if SLICE_KINDS_CARD_REWARDS != tuple(range(_REWARD_LO, _REWARD_HI + 1)):
        raise RuntimeError(
            "The reward slice kinds must be contiguous and ascending: _embed_identity "
            "range-tests them"
        )


_check_tables()
