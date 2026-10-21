from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.encoding.card import ENCODING_DIM_CARD
from src.rl.encoding.character import ENCODING_DIM_CHARACTER
from src.rl.encoding.effect import ENCODING_DIM_EFFECT
from src.rl.encoding.event import ENCODING_DIM_EVENT_OPTION
from src.rl.encoding.focus import OFF_FOCUS_DRAW_ORDER_VISIBLE
from src.rl.encoding.health_block import ENCODING_DIM_HEALTH_BLOCK
from src.rl.encoding.modifier import ENCODING_DIM_MODIFIER
from src.rl.encoding.monster import ENCODING_DIM_MONSTER
from src.rl.encoding.potion import ENCODING_DIM_POTION
from src.rl.encoding.relic import ENCODING_DIM_RELIC
from src.rl.encoding.shop import ENCODING_DIM_PRICE
from src.rl.packed import rows_of_states
from src.rl.types import NUM_ACTION_TYPES
from src.rl.types import SliceKind
from src.rl.types import X_DrawOrderStates
from src.rl.types import X_EffectPending
from src.rl.types import X_EntitySparse
from src.rl.types import X_GameState
from src.rl.types import X_OwnedSparse
from src.rl.types import X_Rooms
from src.rl.types import X_Window


class EntityProjector(nn.Module):
    """Factored (AlphaStar-style) entity embedder: a private first layer per block, the shared
    row families (health/block, prices, effects, modifiers) added into their owners, then one
    shared second layer + LayerNorm. Returns the window packed: its tokens (the six blocks then
    the rooms, state after state) and their X_Window; then the draw-order layer's states, and the
    embedded pending halt."""

    def __init__(self, dim_entity: int, dim_map: int) -> None:
        super().__init__()

        self._dim_entity = dim_entity

        # Per-Entity kind private mappers
        self._pre_card = nn.Linear(ENCODING_DIM_CARD, dim_entity)
        self._pre_relic = nn.Linear(ENCODING_DIM_RELIC, dim_entity)
        self._pre_potion = nn.Linear(ENCODING_DIM_POTION, dim_entity)
        self._pre_event_option = nn.Linear(ENCODING_DIM_EVENT_OPTION, dim_entity)
        self._pre_monster = nn.Linear(ENCODING_DIM_MONSTER, dim_entity)
        self._pre_character = nn.Linear(ENCODING_DIM_CHARACTER, dim_entity)
        self._pre_room = nn.Linear(dim_map, dim_entity)  # Rooms come processed by `MapGNN`

        # Health and Block's mapper is shared between the Character and the Monsters
        self._pre_health_block = nn.Linear(ENCODING_DIM_HEALTH_BLOCK, dim_entity, bias=False)

        # Prices' mapper is shared between the Cards, Relics and Potions
        self._pre_price = nn.Linear(ENCODING_DIM_PRICE, dim_entity, bias=False)

        # Per-Effect embedder w/ >=2 layers is a must due to OHE-summing semantics
        self._effects_mlp = nn.Sequential(
            nn.Linear(ENCODING_DIM_EFFECT, 2 * dim_entity),
            nn.ReLU(),
            nn.Linear(2 * dim_entity, dim_entity, bias=False),
        )

        # Per-Modifier embedder, same shape / logic as the Effects'
        self._modifiers_mlp = nn.Sequential(
            nn.Linear(ENCODING_DIM_MODIFIER, 2 * dim_entity),
            nn.ReLU(),
            nn.Linear(2 * dim_entity, dim_entity, bias=False),
        )

        # Shared second layer (after ReLU, on the fused all-blocks row tensor)
        self._post_shared = nn.Linear(dim_entity, dim_entity)

        # LayerNorm cuz LayerNorm
        self._layer_norm = nn.LayerNorm(dim_entity)

    def forward(
        self, x_gs: X_GameState, x_rooms: torch.Tensor
    ) -> tuple[torch.Tensor, X_Window, X_DrawOrderStates, torch.Tensor]:
        # Shapes: B = batch, D = embedding width, N_tok = the window's tokens, P = the states
        # with a pending halt, A = the action types

        # X_GameState order: the list below IS the window's entity order, block after block
        # within each state. Per block: its rows, each state's count of them, and the rows through
        # the block's private first layer
        window_blocks: list[tuple[X_EntitySparse | X_Rooms, torch.Tensor]] = [
            (x_gs.cards, _project_sparse(self._pre_card, x_gs.cards)),
            (x_gs.relics, _project_sparse(self._pre_relic, x_gs.relics)),
            (x_gs.potions, _project_sparse(self._pre_potion, x_gs.potions)),
            (x_gs.monsters, _project_sparse(self._pre_monster, x_gs.monsters)),
            (x_gs.event_options, _project_sparse(self._pre_event_option, x_gs.event_options)),
            (x_gs.character, _project_sparse(self._pre_character, x_gs.character)),
            (x_gs.rooms, self._pre_room(x_rooms)),
        ]

        # Each state's tokens of each block, and in all; its first row, and its first row of each
        # block
        x_count_blocks = torch.stack([x_block.x_count for x_block, _ in window_blocks], 1) # (B, 7)
        x_count_window = x_count_blocks.sum(dim=1)  # (B,)
        x_first_row = x_count_window.cumsum(0) - x_count_window  # (B,)
        x_first_row_blocks = x_first_row.unsqueeze(1) + x_count_blocks.cumsum(1) - x_count_blocks
        batch_size = x_count_window.shape[0]

        # Each block's tokens written at their rows of the window, with their slice kinds and
        # masks
        x_block_row = torch.cat(
            [
                rows_of_states(
                    x_first_row_blocks[:, b], x_block.x_count, x_block.x_slice_kind.shape[0]
                )
                for b, (x_block, _) in enumerate(window_blocks)
            ]
        )  # (N_tok,)
        num_tokens = x_block_row.shape[0]
        x_tokens = x_rooms.new_empty(num_tokens, self._dim_entity)  # (N_tok, D)
        x_tokens[x_block_row] = torch.cat([x_first for _, x_first in window_blocks])
        x_slice_kind = x_block_row.new_empty(num_tokens)  # (N_tok,)
        x_slice_kind[x_block_row] = torch.cat(
            [x_block.x_slice_kind for x_block, _ in window_blocks]
        ).long()
        x_legal = x_block_row.new_empty(num_tokens, NUM_ACTION_TYPES, dtype=torch.bool)
        x_legal[x_block_row] = torch.cat([x_block.x_legal for x_block, _ in window_blocks])
        x_needs_target = x_block_row.new_empty(num_tokens, dtype=torch.bool)  # (N_tok,)
        x_needs_target[x_block_row] = torch.cat(
            [x_block.x_needs_target for x_block, _ in window_blocks]
        )
        x_state = torch.repeat_interleave(x_count_window, output_size=num_tokens)  # (N_tok,)
        x_slice_kind_count, x_slice_kind_row = _slice_kind_tables(
            x_state, x_slice_kind, batch_size
        )

        # The draw-order layer's states, whose draw pile shows an order (2 or more draw tokens):
        # `x_ordered` masks those under Frozen Eye (visible), then the others (hidden)
        x_count_pile = x_slice_kind_count[:, SliceKind.CARD_PILE_DRAW]  # (B,)
        x_draw_order_visible = x_gs.focus[:, OFF_FOCUS_DRAW_ORDER_VISIBLE] > 0
        x_ordered = (x_count_pile >= 2) & torch.stack(
            [x_draw_order_visible, ~x_draw_order_visible]
        )  # (2, B)

        # The layer's sizes in one read: the CPU waits for the GPU here, and nowhere else; every
        # other size is a shape
        (
            num_states_visible,
            num_states_hidden,
            num_rows_visible,
            num_rows_hidden,
            num_pile_visible,
            num_pile_hidden,
        ) = torch.cat(
            [
                x_ordered.sum(dim=1),
                (x_ordered * x_count_window).sum(dim=1),
                (x_ordered * x_count_pile).sum(dim=1),
            ]
        ).tolist()
        x_draw_order = X_DrawOrderStates(
            x_states=torch.cat(
                [
                    torch.nonzero_static(x_ordered[0], size=num_states_visible),
                    torch.nonzero_static(x_ordered[1], size=num_states_hidden),
                ]
            ).squeeze(1),
            num_states_visible=num_states_visible,
            num_rows_visible=num_rows_visible,
            num_rows=num_rows_visible + num_rows_hidden,
            num_pile_visible=num_pile_visible,
            num_pile=num_pile_visible + num_pile_hidden,
        )

        # Owned rows add into their owners, scaled by 1/sqrt(rows per owner), before the shared
        # ReLU/LayerNorm. Family order matters: float sums are order-sensitive
        _add_owned(x_tokens, x_gs.prices, self._pre_price, x_slice_kind_row)
        _add_owned(x_tokens, x_gs.health_block, self._pre_health_block, x_slice_kind_row)
        _add_owned(x_tokens, x_gs.effects, self._effects_mlp, x_slice_kind_row)
        _add_owned(x_tokens, x_gs.modifiers, self._modifiers_mlp, x_slice_kind_row)

        # The pending halt owns no token: its effect row is a global-context input instead
        x_effect_pending = x_gs.effect_pending
        x_pending_state = torch.repeat_interleave(
            x_effect_pending.x_count, output_size=x_effect_pending.x_column.shape[0]
        )  # (P,)
        x_pending_valid = _project_sparse(self._effects_mlp, x_effect_pending)  # (P, D)
        x_pending = x_pending_valid.new_zeros(batch_size, self._dim_entity).index_copy(
            0, x_pending_state, x_pending_valid
        )  # (B, D)

        # Shared second layer + norm on the rows
        x_hidden = self._layer_norm(self._post_shared(F.relu(x_tokens)))  # (N_tok, D)

        x_window = X_Window(
            x_slice_kind=x_slice_kind,
            x_legal=x_legal,
            x_needs_target=x_needs_target,
            x_state=x_state,
            x_block_row=x_block_row,
            x_first_row=x_first_row,
            x_count=x_count_window,
            x_slice_kind_row=x_slice_kind_row,
            x_slice_kind_count=x_slice_kind_count,
        )
        return (
            x_hidden,
            x_window,
            x_draw_order,
            x_pending,
        )


def _add_pooled(x_target: torch.Tensor, x_pos: torch.Tensor, x_rows: torch.Tensor) -> None:
    """Add each row into `x_target[pos]`, scaled by 1/sqrt(the rows sharing its pos)."""

    # Counted at the target's length: bincount reads max(pos) back to size its output
    x_count = torch.zeros(x_target.shape[0], dtype=torch.long, device=x_pos.device)
    x_count = x_count.index_add_(0, x_pos, torch.ones_like(x_pos))[x_pos]  # (N,)
    x_target.index_add_(0, x_pos, x_rows / x_count.to(x_rows.dtype).sqrt().unsqueeze(1))


class _SumSparseDenseBackward(torch.autograd.Function):
    """F.embedding_bag over the sparse entries, with the weight's gradient as one GEMM over the
    entries densified to (N, in) rows: embedding_bag's own backward sorts the entries and sums
    each column's serially"""

    @staticmethod
    def forward(
        ctx: Any, x_column: torch.Tensor, x_value: torch.Tensor, x_weight: torch.Tensor
    ) -> torch.Tensor:
        """The (N, K) entries through a linear layer's (D, in) weight, bias aside: its columns at
        the entries' columns, scaled by their values, summed"""
        ctx.save_for_backward(x_column, x_value)
        ctx.num_columns = x_weight.shape[1]

        # The transposed weight copied whole: as a strided view, a GPU reads its rows uncoalesced
        return F.embedding_bag(
            x_column.int(), x_weight.t().contiguous(), per_sample_weights=x_value, mode="sum"
        )

    @staticmethod
    def backward(ctx: Any, x_grad: torch.Tensor) -> tuple[None, None, torch.Tensor]:
        x_column, x_value = ctx.saved_tensors

        # The pad entries (0, 0.0) add zeros
        x_dense = x_grad.new_zeros(x_column.shape[0], ctx.num_columns)
        x_dense.scatter_add_(1, x_column.long(), x_value)  # (N, in)
        return None, None, x_grad.t() @ x_dense  # (D, in)


def _project_sparse(
    mapper: nn.Linear | nn.Sequential, x_block: X_EntitySparse | X_OwnedSparse | X_EffectPending
) -> torch.Tensor:
    """`mapper` on the dense rows that `x_block`'s (N, K) entries stand for. Its first linear layer
    (the mapper, or a Sequential's first layer) sums its weight's columns at the entries' columns,
    scaled by their values, plus its bias; a Sequential's other layers follow"""
    linear = mapper if isinstance(mapper, nn.Linear) else mapper[0]
    x_rows = _SumSparseDenseBackward.apply(x_block.x_column, x_block.x_value, linear.weight)
    if linear.bias is not None:
        x_rows = x_rows + linear.bias
    return x_rows if isinstance(mapper, nn.Linear) else mapper[1:](x_rows)


def _add_owned(
    x_tokens: torch.Tensor,
    x_owned_sparse: X_OwnedSparse,
    mapper: nn.Linear | nn.Sequential,
    x_slice_kind_row: torch.Tensor,
) -> None:
    """Add a family's rows, each state's `x_count` of them, into their owners' rows of `x_tokens`.
    An owner is the rank-th token of its slice kind, whose tokens sit in consecutive rows"""
    x_state = torch.repeat_interleave(
        x_owned_sparse.x_count, output_size=x_owned_sparse.x_column.shape[0]
    )
    x_rows = _project_sparse(mapper, x_owned_sparse)  # (N, D)
    x_owner_first_row = x_slice_kind_row[x_state, x_owned_sparse.x_owner_slice_kind]  # (N,)
    _add_pooled(x_tokens, x_owner_first_row + x_owned_sparse.x_owner_rank, x_rows)


def _slice_kind_tables(
    x_state: torch.Tensor, x_slice_kind: torch.Tensor, num_states: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each state's tokens of each slice kind, and its row of each slice kind's first token, both
    (B, len(SliceKind)), from each row's state and slice kind: the window's rows (out of range)
    where a slice kind is absent. Each block keeps a slice kind's tokens in consecutive rows"""
    num_rows = x_state.shape[0]
    x_key = x_state * len(SliceKind) + x_slice_kind  # (N_tok,) each row's (state, slice kind)
    x_slice_kind_count = x_key.new_zeros(num_states * len(SliceKind))
    x_slice_kind_count.index_add_(0, x_key, torch.ones_like(x_key))
    x_slice_kind_row = x_key.new_full((num_states * len(SliceKind),), num_rows)
    x_slice_kind_row.scatter_reduce_(
        0, x_key, torch.arange(num_rows, device=x_key.device), reduce="amin"
    )
    return x_slice_kind_count.view(num_states, -1), x_slice_kind_row.view(num_states, -1)
