from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.encoding.effect import ENCODING_DIM_EFFECT
from src.rl.encoding.focus import OFF_FOCUS_DRAW_ORDER_VISIBLE
from src.rl.encoding.modifier import ENCODING_DIM_MODIFIER
from src.rl.encoding.state import ENCODING_DIM_TOKEN
from src.rl.types import TOKEN_KINDS_OF_FAMILY
from src.rl.types import TokenKind
from src.rl.types import X_DrawOrderStates
from src.rl.types import X_EffectPending
from src.rl.types import X_GameState
from src.rl.types import X_OwnedSparse
from src.rl.types import X_Tokens
from src.rl.types import X_Window


# Each token kind's family: its index in TOKEN_KINDS_OF_FAMILY
_FAMILY_OF_TOKEN_KIND = {
    token_kind: family
    for family, token_kinds in enumerate(TOKEN_KINDS_OF_FAMILY)
    for token_kind in token_kinds
}


class EntityProjector(nn.Module):
    """Factored (AlphaStar-style) entity embedder: one first layer over the shared columns plus
    each token's family bias, the rooms' map nodes through their own, the shared row families
    (effects, modifiers) added into their owners, then one shared second layer + LayerNorm.
    Returns the window packed: its tokens, every state's in window order, state after state, and
    their X_Window; then the draw-order layer's states, and the embedded pending halt."""

    # The buffer __init__ registers, typed for mypy
    _family_of_token_kind: torch.Tensor

    def __init__(self, dim_entity: int, dim_map: int) -> None:
        super().__init__()

        self._dim_entity = dim_entity

        # Every family's first layer: one projection over the shared columns, then the family's
        # bias
        self._pre_token = nn.Linear(ENCODING_DIM_TOKEN, dim_entity, bias=False)
        self._bias_family = nn.Parameter(torch.zeros(len(TOKEN_KINDS_OF_FAMILY), dim_entity))
        self._pre_room = nn.Linear(dim_map, dim_entity)  # Rooms come processed by `MapGNN`

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

        # Shared second layer (after ReLU, on the token rows)
        self._post_shared = nn.Linear(dim_entity, dim_entity)

        # LayerNorm cuz LayerNorm
        self._layer_norm = nn.LayerNorm(dim_entity)

        # Each token kind's family; derived, so excluded from checkpoints
        self.register_buffer(
            "_family_of_token_kind",
            torch.tensor([_FAMILY_OF_TOKEN_KIND[token_kind] for token_kind in TokenKind]),
            persistent=False,
        )

    def forward(
        self, x_gs: X_GameState, x_hidden_map: torch.Tensor, x_map_of_state: torch.Tensor
    ) -> tuple[torch.Tensor, X_Window, X_DrawOrderStates, torch.Tensor]:
        """`x_gs`'s window, with the map GNN's per-node features of each unique map `x_hidden_map`
        (U, H * W, dim_map) and each state's map `x_map_of_state` (B,)"""
        # Shapes: B = batch, D = embedding width, N_tok = the window's tokens, P = the states
        # with a pending halt, A = the action types

        # Each token's kind, family and state, and each state's tokens
        x_token_kind = x_gs.tokens.x_token_kind.long()  # (N_tok,)
        x_family = self._family_of_token_kind[x_token_kind]  # (N_tok,)
        x_count_window = x_gs.tokens.x_count  # (B,)
        x_state = torch.repeat_interleave(x_count_window, output_size=x_token_kind.shape[0])
        batch_size = x_count_window.shape[0]
        x_token_kind_count, x_token_kind_row = _token_kind_tables(
            x_state, x_token_kind, batch_size
        )

        # First layer: the shared columns' projection, then each token's family bias
        x_first = _project_sparse(self._pre_token, x_gs.tokens) + SpreadEmbedding.apply(
            self._bias_family, x_family
        )

        # The draw-order layer's states, whose draw pile shows an order (2 or more draw tokens):
        # `x_ordered` masks those under Frozen Eye (visible), then the others (hidden)
        x_count_pile = x_token_kind_count[:, TokenKind.CARD_PILE_DRAW]  # (B,)
        x_draw_order_visible = x_gs.focus[:, OFF_FOCUS_DRAW_ORDER_VISIBLE] > 0
        x_ordered = (x_count_pile >= 2) & torch.stack(
            [x_draw_order_visible, ~x_draw_order_visible]
        )  # (2, B)

        # The layer's sizes and each family's tokens in one read: the CPU waits for the GPU here,
        # and nowhere else; every other size is a shape
        (
            num_states_visible,
            num_states_hidden,
            num_rows_visible,
            num_rows_hidden,
            num_pile_visible,
            num_pile_hidden,
            *num_tokens_family,
        ) = torch.cat(
            [
                x_ordered.sum(dim=1),
                (x_ordered * x_count_window).sum(dim=1),
                (x_ordered * x_count_pile).sum(dim=1),
                x_family.new_zeros(len(TOKEN_KINDS_OF_FAMILY)).index_add_(
                    0, x_family, torch.ones_like(x_family)
                ),
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

        # Rooms: each room token's node in its state's map, through _pre_room in place of the first
        # layer (no entries, no family bias)
        num_rooms = num_tokens_family[_FAMILY_OF_TOKEN_KIND[TokenKind.ROOMS]]
        x_room_row = torch.nonzero_static(x_token_kind == TokenKind.ROOMS, size=num_rooms)
        x_room_row = x_room_row.squeeze(1)  # (N_rooms,)
        x_node_hidden = x_hidden_map[
            x_map_of_state[x_state[x_room_row]], x_gs.tokens.x_node_idx[x_room_row].long()
        ]  # (N_rooms, dim_map)
        x_first[x_room_row] = self._pre_room(x_node_hidden)

        # Owned rows add into their owners, scaled by 1/sqrt(rows per owner), before the shared
        # ReLU/LayerNorm
        _add_owned(x_first, x_gs.tokens.effects, self._effects_mlp)
        _add_owned(x_first, x_gs.tokens.modifiers, self._modifiers_mlp)

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
        x_hidden = self._layer_norm(self._post_shared(F.relu(x_first)))  # (N_tok, D)

        x_window = X_Window(
            x_token_kind=x_token_kind,
            x_family=x_family,
            x_legal=x_gs.tokens.x_legal,
            x_needs_target=x_gs.tokens.x_needs_target,
            x_state=x_state,
            x_count=x_count_window,
            x_token_kind_row=x_token_kind_row,
            x_token_kind_count=x_token_kind_count,
            num_tokens_family=num_tokens_family,
        )
        return (
            x_hidden,
            x_window,
            x_draw_order,
            x_pending,
        )


# The copies of a table that SpreadEmbedding's backward sums over
_SPREAD_COPIES = 256


class SpreadEmbedding(torch.autograd.Function):
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
    mapper: nn.Linear | nn.Sequential, x_block: X_Tokens | X_OwnedSparse | X_EffectPending
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
    x_first: torch.Tensor, x_owned_sparse: X_OwnedSparse, mapper: nn.Sequential
) -> None:
    """Add a family's rows, each token's `x_count` of them, into their owners' rows of `x_first`,
    scaled by 1/sqrt(the owner's count)"""
    x_count = x_owned_sparse.x_count  # (N_tok,) int16

    # Each row's owner, from the counts widened: repeat_interleave takes no int16
    x_owner = torch.repeat_interleave(
        x_count.long(), output_size=x_owned_sparse.x_column.shape[0]
    )  # (M,)
    x_rows = _project_sparse(mapper, x_owned_sparse)  # (M, D)
    x_first.index_add_(0, x_owner, x_rows / x_count[x_owner].to(x_rows.dtype).sqrt().unsqueeze(1))


def _token_kind_tables(
    x_state: torch.Tensor, x_token_kind: torch.Tensor, num_states: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each state's tokens of each token kind, and its row of each token kind's first token, both
    (B, len(TokenKind)), from each row's state and token kind: the window's rows (out of range)
    where a token kind is absent. A token kind's tokens sit in consecutive rows"""
    num_rows = x_state.shape[0]
    x_key = x_state * len(TokenKind) + x_token_kind  # (N_tok,) each row's (state, token kind)
    x_token_kind_count = x_key.new_zeros(num_states * len(TokenKind))
    x_token_kind_count.index_add_(0, x_key, torch.ones_like(x_key))
    x_token_kind_row = x_key.new_full((num_states * len(TokenKind),), num_rows)
    x_token_kind_row.scatter_reduce_(
        0, x_key, torch.arange(num_rows, device=x_key.device), reduce="amin"
    )
    return x_token_kind_count.view(num_states, -1), x_token_kind_row.view(num_states, -1)
