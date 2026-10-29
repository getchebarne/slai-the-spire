from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.encoding.effect import ENCODING_DIM_EFFECT
from src.rl.encoding.focus import OFF_FOCUS_DRAW_ORDER_VISIBLE
from src.rl.encoding.modifier import ENCODING_DIM_MODIFIER
from src.rl.encoding.state import ENCODING_DIM_TOKEN
from src.rl.models.spread_embedding import SpreadEmbedding
from src.rl.types import TOKEN_KINDS_OF_ENTITY_KIND
from src.rl.types import TokenKind
from src.rl.types import X_DrawOrderStates
from src.rl.types import X_EffectPending
from src.rl.types import X_GameState
from src.rl.types import X_OwnedSparse
from src.rl.types import X_Tokens
from src.rl.types import X_Window


# Each token kind's Entity kind: its index in `TOKEN_KINDS_OF_ENTITY_KIND`
_MAP_TOKEN_KIND_TO_ENTITY_KIND = {
    token_kind: entity_kind
    for entity_kind, token_kinds in enumerate(TOKEN_KINDS_OF_ENTITY_KIND)
    for token_kind in token_kinds
}


class EntityProjector(nn.Module):
    """Factored (AlphaStar-style) Entity embedder.

    One first layer over the shared columns plus each token's Entity kind bias, the Rooms' Map
    nodes through their own, the owned row kinds (Effects, Modifiers) added into their owners,
    then one shared second layer + LayerNorm. Returns the window packed: its tokens, every
    state's in window order, state after state, and their X_Window; then the draw-order layer's
    states, and the embedded pending halt.

    Parameters
    ----------
    dim_entity : int
        The width of every Entity token's embedding.
    dim_map : int
        The width of the Map GNN's node features.
    """

    # The buffer `__init__` registers, typed for mypy
    _entity_kind_of_token_kind: torch.Tensor

    def __init__(self, dim_entity: int, dim_map: int) -> None:
        super().__init__()

        self._dim_entity = dim_entity

        # Every Entity kind's first layer: one projection over the shared columns
        self._pre_token = nn.Linear(ENCODING_DIM_TOKEN, dim_entity, bias=False)

        # Per-Entity kind bias
        self._bias_entity_kind = nn.Parameter(
            torch.zeros(len(TOKEN_KINDS_OF_ENTITY_KIND), dim_entity)
        )

        # Rooms are separate because they come already-processed by `MapGNN`
        self._pre_room = nn.Linear(dim_map, dim_entity)

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

        # Each token kind's Entity kind; derived, so excluded from checkpoints
        self.register_buffer(
            "_entity_kind_of_token_kind",
            torch.tensor([_MAP_TOKEN_KIND_TO_ENTITY_KIND[token_kind] for token_kind in TokenKind]),
            persistent=False,
        )

    def forward(
        self, x_gs: X_GameState, x_hidden_map: torch.Tensor, x_map_of_state: torch.Tensor
    ) -> tuple[torch.Tensor, X_Window, X_DrawOrderStates, torch.Tensor]:
        """Embed `x_gs`'s window.

        Parameters
        ----------
        x_gs : X_GameState
            The game states.
        x_hidden_map : torch.Tensor
            (U, H * W, dim_map). The Map GNN's per-node features of each Map it ran on (see
            `MapGNN.forward`).
        x_map_of_state : torch.Tensor
            (B,) long. Each state's Map, its index in `x_hidden_map`.

        Returns
        -------
        x_tokens : torch.Tensor
            (N_tok, D). The window's tokens, every state's in window order, state after state.
        x_window : X_Window
            The tokens' X_Window.
        x_draw_order : X_DrawOrderStates
            The draw-order layer's states.
        x_pending : torch.Tensor
            (B, D). Each state's embedded pending halt, zero where none is pending.
        """
        # Shapes: B = batch, D = embedding width, N_tok = the window's tokens, P = the states
        # with a pending halt, A = the action types
        batch_size = x_gs.tokens.x_count.shape[0]
        num_tokens = x_gs.tokens.x_token_kind.shape[0]

        # Each token's token kind, Entity kind and state, and each state's tokens
        x_token_kind = x_gs.tokens.x_token_kind.long()  # (N_tok,)
        x_entity_kind = self._entity_kind_of_token_kind[x_token_kind]  # (N_tok,)

        # Handy alias
        x_count_window = x_gs.tokens.x_count  # (B,)

        # First layer: the shared columns' projection, then each token's Entity kind bias
        x_first = _project_sparse(self._pre_token, x_gs.tokens) + SpreadEmbedding.apply(
            self._bias_entity_kind, x_entity_kind
        )

        # State identifier for each token in the batch
        x_state = torch.repeat_interleave(x_count_window, output_size=num_tokens)  # (N_tok,)

        # `x_token_kind_count`:   Number of tokens of each `TokenKind` for each state
        # `x_token_kind_index`:   Index of first token of each `TokenKind` for each state
        x_token_kind_count, x_token_kind_row = _token_kind_tables(
            x_state, x_token_kind, batch_size
        )

        # The draw-order layer's states, whose draw pile shows an order (2 or more draw tokens):
        # `x_ordered` masks those under Frozen Eye (visible), then the others (hidden)
        x_count_pile = x_token_kind_count[:, TokenKind.CARD_PILE_DRAW]  # (B,)
        x_draw_order_visible = x_gs.focus[:, OFF_FOCUS_DRAW_ORDER_VISIBLE] > 0
        x_ordered = (x_count_pile >= 2) & torch.stack(
            [x_draw_order_visible, ~x_draw_order_visible]
        )  # (2, B)

        # The layer's sizes and each Entity kind's tokens in one read: the CPU waits for the GPU
        # here, and nowhere else; every other size is a shape
        (
            num_states_visible,
            num_states_hidden,
            num_rows_visible,
            num_rows_hidden,
            num_pile_visible,
            num_pile_hidden,
            *num_tokens_entity_kind,
        ) = torch.cat(
            [
                x_ordered.sum(dim=1),
                (x_ordered * x_count_window).sum(dim=1),
                (x_ordered * x_count_pile).sum(dim=1),
                torch.zeros(
                    len(TOKEN_KINDS_OF_ENTITY_KIND), dtype=torch.long, device=x_entity_kind.device
                ).index_add_(0, x_entity_kind, torch.ones_like(x_entity_kind)),
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

        # Rooms: each Room token's node in its state's Map, through _pre_room in place of the first
        # layer (no entries, no Entity kind bias)
        num_rooms = num_tokens_entity_kind[_MAP_TOKEN_KIND_TO_ENTITY_KIND[TokenKind.ROOMS]]
        x_room_row = torch.nonzero_static(x_token_kind == TokenKind.ROOMS, size=num_rooms)
        x_room_row = x_room_row.squeeze(1)  # (N_rooms,)
        x_node_hidden = x_hidden_map[
            x_map_of_state[x_state[x_room_row]], x_gs.tokens.x_node_idx[x_room_row].long()
        ]  # (N_rooms, dim_map)
        x_first[x_room_row] = self._pre_room(x_node_hidden)

        # Owned rows add into their owners, scaled by 1/sqrt(rows per owner), before the shared
        # ReLU/LayerNorm
        _add_owned(x_first, x_gs.tokens.effect_rows, self._effects_mlp)
        _add_owned(x_first, x_gs.tokens.modifier_rows, self._modifiers_mlp)

        # The pending halt owns no token: its Effect row is a global-context input instead
        x_effect_pending = x_gs.effect_pending
        x_pending_state = torch.repeat_interleave(
            x_effect_pending.x_count, output_size=x_effect_pending.x_column.shape[0]
        )  # (P,)
        x_pending_valid = _project_sparse(self._effects_mlp, x_effect_pending)  # (P, D)
        x_pending = torch.zeros(
            batch_size,
            self._dim_entity,
            dtype=x_pending_valid.dtype,
            device=x_pending_valid.device,
        )
        x_pending = x_pending.index_copy(0, x_pending_state, x_pending_valid)  # (B, D)

        # Shared second layer + norm on the rows
        x_hidden = self._layer_norm(self._post_shared(F.relu(x_first)))  # (N_tok, D)

        x_window = X_Window(
            x_token_kind=x_token_kind,
            x_entity_kind=x_entity_kind,
            x_legal=x_gs.tokens.x_legal,
            x_needs_target=x_gs.tokens.x_needs_target,
            x_state=x_state,
            x_count=x_count_window,
            x_token_kind_row=x_token_kind_row,
            x_token_kind_count=x_token_kind_count,
            num_tokens_entity_kind=num_tokens_entity_kind,
        )
        return (
            x_hidden,
            x_window,
            x_draw_order,
            x_pending,
        )


def _project_sparse(
    mapper: nn.Linear | nn.Sequential, x_block: X_Tokens | X_OwnedSparse | X_EffectPending
) -> torch.Tensor:
    """Run `mapper` on the dense rows that `x_block`'s (N, K) entries stand for.

    Its first linear layer (the mapper, or a Sequential's first layer) sums its weight's columns
    at the entries' columns, scaled by their values, plus its bias; a Sequential's other layers
    follow.

    Parameters
    ----------
    mapper : nn.Linear | nn.Sequential
        A linear layer, or a Sequential whose first layer is linear.
    x_block : X_Tokens | X_OwnedSparse | X_EffectPending
        The block, its rows held sparse.

    Returns
    -------
    torch.Tensor
        (N, D). `mapper`'s output on each row.
    """
    linear = mapper if isinstance(mapper, nn.Linear) else mapper[0]
    x_rows = _SumSparseDenseBackward.apply(x_block.x_column, x_block.x_value, linear.weight)
    if linear.bias is not None:
        x_rows = x_rows + linear.bias
    return x_rows if isinstance(mapper, nn.Linear) else mapper[1:](x_rows)


def _add_owned(
    x_first: torch.Tensor, x_owned_sparse: X_OwnedSparse, mapper: nn.Sequential
) -> None:
    """Add an owned row kind's rows into their owners' rows of `x_first`.

    Each token owns `x_count` of the rows, and each row is scaled by 1/sqrt(its owner's count).

    Parameters
    ----------
    x_first : torch.Tensor
        (N_tok, D). The tokens' first-layer rows, updated in place.
    x_owned_sparse : X_OwnedSparse
        The owned row kind's rows, held sparse.
    mapper : nn.Sequential
        The owned row kind's MLP.
    """
    x_count = x_owned_sparse.x_count  # (N_tok,) int16

    # Each row's owner, from the counts widened: repeat_interleave takes no int16
    x_owner = torch.repeat_interleave(
        x_count.long(), output_size=x_owned_sparse.x_column.shape[0]
    )  # (M,)
    x_rows = _project_sparse(mapper, x_owned_sparse)  # (M, D)
    x_first.index_add_(0, x_owner, x_rows / x_count[x_owner].to(x_rows.dtype).sqrt().unsqueeze(1))


def _token_kind_tables(
    x_state: torch.Tensor, x_token_kind: torch.Tensor, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count each state's tokens of each token kind and find each token kind's first row.

    A token kind's tokens sit in consecutive rows.

    Parameters
    ----------
    x_state : torch.Tensor
        (N_tok,) long. Each row's state.
    x_token_kind : torch.Tensor
        (N_tok,) long. Each row's token kind.
    batch_size : int
        The number of states, B.

    Returns
    -------
    x_token_kind_count : torch.Tensor
        (B, len(TokenKind)) long. Each state's tokens of each token kind.
    x_token_kind_row : torch.Tensor
        (B, len(TokenKind)) long. Each state's row of each token kind's first token; the
        window's rows (out of range) where the token kind is absent.
    """
    num_tokens = x_state.shape[0]

    # Initialize tensors to:
    #     1. Store the amount of tokens for each kind in each state
    #     2. Store the index of the first token for each token kind in each state
    x_token_kind_count = torch.zeros(
        (batch_size * len(TokenKind),), dtype=torch.long, device=x_state.device
    )
    x_token_kind_row = torch.full(
        (batch_size * len(TokenKind),), num_tokens, dtype=torch.long, device=x_state.device
    )

    # Calculate the key: each token's state and token kind
    x_key = x_state * len(TokenKind) + x_token_kind  # (N_tok,)

    # Add it for the counts (`x_token_kind_count` is initialized as zeros)
    x_token_kind_count.index_add_(0, x_key, torch.ones_like(x_key))

    # Scatter-reduce to get the index of the first token for each state and token kind
    # Equivalent to:
    # ```
    # for i in range(num_tokens):
    #     cell = x_key[i]
    #     x_token_kind_row[cell] = min(x_token_kind_row[cell], i)
    # ```
    x_token_kind_row.scatter_reduce_(
        0, x_key, torch.arange(num_tokens, device=x_key.device), reduce="amin"
    )

    # Return as (B, len(TokenKind)) pair
    return x_token_kind_count.view(batch_size, -1), x_token_kind_row.view(batch_size, -1)


class _SumSparseDenseBackward(torch.autograd.Function):
    """`F.embedding_bag` over the sparse entries, with the weight's gradient as one GEMM.

    The GEMM runs over the entries densified to (N, in) rows: `embedding_bag`'s own backward sorts
    the entries and sums each column's serially.
    """

    @staticmethod
    def forward(
        ctx: Any, x_column: torch.Tensor, x_value: torch.Tensor, x_weight: torch.Tensor
    ) -> torch.Tensor:
        """Run the (N, K) entries through a linear layer's weight, bias aside.

        Sums the weight's columns at the entries' columns, scaled by their values.

        Parameters
        ----------
        x_column : torch.Tensor
            (N, K) int16. The entries' columns.
        x_value : torch.Tensor
            (N, K) float32. The entries' values.
        x_weight : torch.Tensor
            (D, in). The linear layer's weight.

        Returns
        -------
        torch.Tensor
            (N, D). Each row's sum.
        """
        ctx.save_for_backward(x_column, x_value)
        ctx.num_columns = x_weight.shape[1]

        # The transposed weight copied whole: as a strided view, a GPU reads its rows uncoalesced
        return F.embedding_bag(
            x_column.int(), x_weight.t().contiguous(), per_sample_weights=x_value, mode="sum"
        )

    @staticmethod
    def backward(ctx: Any, x_grad: torch.Tensor) -> tuple[None, None, torch.Tensor]:
        """Take the weight's gradient as one GEMM over the entries densified to (N, in) rows.

        Parameters
        ----------
        x_grad : torch.Tensor
            (N, D). Each row's output gradient.

        Returns
        -------
        None
            The entries' columns take no gradient.
        None
            The entries' values take no gradient.
        torch.Tensor
            (D, in). The weight's gradient.
        """
        x_column, x_value = ctx.saved_tensors

        # The pad entries (0, 0.0) add zeros
        x_dense = torch.zeros(
            x_column.shape[0], ctx.num_columns, dtype=x_grad.dtype, device=x_grad.device
        )
        x_dense.scatter_add_(1, x_column.long(), x_value)  # (N, in)
        return None, None, x_grad.t() @ x_dense  # (D, in)
