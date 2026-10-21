from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.encoding.energy import ENCODING_DIM_ENERGY
from src.rl.encoding.event import ENCODING_DIM_EVENT_META
from src.rl.encoding.focus import ENCODING_DIM_FOCUS
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
from src.rl.types import X_GameState


# Count features: one per slice kind except the character singleton (a constant 1.0), each
# normalized by its slice kind's cap (a count never exceeds it), in window order
_COUNT_SLICE_KINDS = tuple(
    slice_kind for slice_kind in SliceKind if slice_kind is not SliceKind.CHARACTER
)

# The copies of a table that _SpreadEmbedding's backward sums over
_SPREAD_COPIES = 256


class _SpreadEmbedding(torch.autograd.Function):
    """F.embedding, whose backward sums each table row's gradient over _SPREAD_COPIES copies of
    the table: the token at flat position i adds into copy i % _SPREAD_COPIES, then the copies
    add up. The same sums in another order, in short loops"""

    @staticmethod
    def forward(ctx: Any, x_weight: torch.Tensor, x_idxs: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x_idxs)
        ctx.num_embeddings = x_weight.shape[0]
        return F.embedding(x_idxs, x_weight)

    @staticmethod
    def backward(ctx: Any, x_grad: torch.Tensor) -> tuple[torch.Tensor, None]:
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
        return x_grad_weight, None


class Core(nn.Module):
    # The buffers __init__ registers, typed for mypy
    _count_slice_kinds: torch.Tensor
    _count_norm: torch.Tensor
    _is_reward: torch.Tensor

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
            "_count_slice_kinds", torch.tensor(_COUNT_SLICE_KINDS), persistent=False
        )
        self.register_buffer(
            "_count_norm",
            torch.tensor(
                [float(CAP_OF_SLICE_KIND[slice_kind]) for slice_kind in _COUNT_SLICE_KINDS]
            ).view(1, -1),
            persistent=False,
        )

        # The per-bundle slice kinds carry their own type rows; this shared vector
        # is the global CARD_REWARD identity summed on top of all of them
        self._reward_identity = nn.Parameter(torch.empty(dim_entity))
        nn.init.normal_(self._reward_identity)  # N(0, 1), as the type rows

        # Per slice kind, whether it is a reward bundle's; derived, so excluded from checkpoints
        self.register_buffer(
            "_is_reward",
            torch.tensor([slice_kind in SLICE_KINDS_CARD_REWARDS for slice_kind in SliceKind]),
            persistent=False,
        )

    def forward(self, x_gs: X_GameState, with_value: bool = True) -> X_CoreOutput:
        """The core's output; `with_value=False` runs the draw-order layer only on the states the
        policy reads, for callers that need no value, and leaves `global_value` the policy's"""

        # Map GNN: next-row room features + whole-graph readout
        x_rooms, x_map_readout = self._map_gnn(
            x_gs.map_grid, x_gs.rooms.x_node_idx, x_gs.rooms.x_count
        )

        # The window packed (its tokens and their X_Window), the draw-order layer's states, and
        # the embedded pending halt
        x_tokens, x_window, x_draw_order, x_pending = self._entity_projector(x_gs, x_rooms)

        # One transformer pass over every state, without the draw depths
        x_refined, x_global_token = self._entity_transformer(
            self._embed_identity(x_tokens, x_window.x_slice_kind), x_window
        )

        # Per-state inputs to the global context: the same for the transformer and the layer, so
        # cat'd once
        x_state_inputs = torch.cat(
            [
                x_map_readout,
                self._get_token_counts(x_window.x_slice_kind_count),
                x_gs.energy,
                x_gs.focus,
                x_gs.ascension,
                x_gs.map_meta,
                x_gs.reward_meta,
                x_gs.shop_meta,
                x_gs.event_meta,
                x_pending,
            ],
            dim=1,
        )

        # The one character token per state, its CHARACTER row
        x_character_row = x_window.x_slice_kind_row[:, SliceKind.CHARACTER]  # (B,)
        x_global = self._global_context(x_refined[x_character_row], x_global_token, x_state_inputs)

        # The layer's states, whose pile shows an order: the Frozen Eye ones, then the rest unless
        # `with_value` is False, with their rows and pile rows
        x_states = x_draw_order.x_states
        num_rows, num_pile = x_draw_order.num_rows, x_draw_order.num_pile
        if not with_value:
            x_states = x_states[: x_draw_order.num_states_visible]
            num_rows, num_pile = x_draw_order.num_rows_visible, x_draw_order.num_pile_visible
        if x_states.shape[0] == 0:
            return X_CoreOutput(
                global_=x_global,
                global_value=x_global,
                x_tokens=x_refined,
                window=x_window,
            )

        # The value head reads the layer on every one of its states
        x_rows, x_refined_layer, x_global_token_layer = self._draw_order_layer(
            x_refined, x_global_token, x_window, x_states, num_rows, num_pile
        )

        # Each layer state's character among the layer's rows: the one CHARACTER row of each of
        # its states, state after state
        x_character_layer = torch.nonzero_static(
            x_window.x_slice_kind[x_rows] == SliceKind.CHARACTER, size=x_states.shape[0]
        ).squeeze(1)
        x_global_layer = self._global_context(
            x_refined_layer[x_character_layer], x_global_token_layer, x_state_inputs[x_states]
        )
        x_global_value = x_global.index_copy(0, x_states, x_global_layer)

        # The policy reads it on the Frozen Eye states: their order-aware rows written back into
        # the window
        num_states_visible = x_draw_order.num_states_visible
        if num_states_visible > 0:
            num_rows_visible = x_draw_order.num_rows_visible
            x_global = x_global.index_copy(
                0, x_states[:num_states_visible], x_global_layer[:num_states_visible]
            )
            x_refined = x_refined.index_copy(
                0, x_rows[:num_rows_visible], x_refined_layer[:num_rows_visible]
            )
        return X_CoreOutput(
            global_=x_global,
            global_value=x_global_value,
            x_tokens=x_refined,
            window=x_window,
        )

    def _global_context(
        self, x_character: torch.Tensor, x_global_token: torch.Tensor, x_state_inputs: torch.Tensor
    ) -> torch.Tensor:
        """The global vector of each state, from its global token, its refined character token
        and its per-state inputs"""
        return self._global_projection(
            torch.cat([x_global_token, x_character, x_state_inputs], dim=1)
        )

    def _embed_identity(self, x_tokens: torch.Tensor, x_slice_kind: torch.Tensor) -> torch.Tensor:
        # Every token's identity comes from its slice kind: the slice kind's type row, plus the
        # shared CARD_REWARD identity on the reward slice kinds
        x_is_reward = self._is_reward[x_slice_kind]
        x_type = _SpreadEmbedding.apply(self._type_emb.weight, x_slice_kind)
        return x_tokens + x_type + x_is_reward.unsqueeze(1) * self._reward_identity

    def _get_token_counts(self, x_slice_kind_count: torch.Tensor) -> torch.Tensor:
        """Per-slice-kind token counts, sqrt-normalized by the slice kind's cap (a count never
        exceeds it, so the ratio is in [0, 1])."""
        x_count = x_slice_kind_count[:, self._count_slice_kinds].float()
        return (x_count / self._count_norm).sqrt()
