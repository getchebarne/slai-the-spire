from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.constants import DEDUP_MIN_BATCH
from src.rl.encoding.effect import ENCODING_DIM_EFFECT
from src.rl.encoding.effect import ENCODING_DIM_SPARSE_EFFECT
from src.rl.encoding.focus import OFF_FOCUS_DRAW_ORDER_VISIBLE
from src.rl.encoding.modifier import ENCODING_DIM_MODIFIER
from src.rl.encoding.modifier import ENCODING_DIM_SPARSE_MODIFIER
from src.rl.encoding.state import ENCODING_DIM_SPARSE_TOKEN
from src.rl.encoding.state import ENCODING_DIM_TOKEN
from src.rl.models.spread_embedding import SpreadEmbedding
from src.rl.types import TOKEN_KINDS_OF_ENTITY_KIND
from src.rl.types import TokenKind
from src.rl.types import X_DrawOrderStates
from src.rl.types import X_GameState
from src.rl.types import X_OwnedSparse
from src.rl.types import X_Tokens
from src.rl.types import X_Window


# The most entries in a row of any block the hash reads
_HASH_MAX_ENTRIES = max(
    ENCODING_DIM_SPARSE_TOKEN, ENCODING_DIM_SPARSE_EFFECT, ENCODING_DIM_SPARSE_MODIFIER
)

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

    # The buffers `__init__` registers, typed for mypy
    _entity_kind_of_token_kind: torch.Tensor
    _hash_weights_entry: torch.Tensor
    _hash_weights_field: torch.Tensor

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

        # The dedup hash's odd weights, from a generator of their own; excluded from checkpoints
        generator = torch.Generator().manual_seed(0)
        self.register_buffer(
            "_hash_weights_entry",
            torch.randint(1 << 62, (2, _HASH_MAX_ENTRIES), generator=generator) | 1,
            persistent=False,
        )
        self.register_buffer(
            "_hash_weights_field",
            torch.randint(1 << 62, (6,), generator=generator) | 1,
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

        # The layer's sizes and each Entity kind's tokens in one read: below the dedup's gate the
        # CPU waits for the GPU here, and nowhere else; every other size is a shape
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

        # Rooms: each Room token's node in its state's Map
        num_rooms = num_tokens_entity_kind[_MAP_TOKEN_KIND_TO_ENTITY_KIND[TokenKind.ROOMS]]
        x_room_row = torch.nonzero_static(x_token_kind == TokenKind.ROOMS, size=num_rooms)
        x_room_row = x_room_row.squeeze(1)  # (N_rooms,)
        x_map_room = x_map_of_state[x_state[x_room_row]]  # (N_rooms,)
        x_node_room = x_gs.tokens.x_node_idx[x_room_row].long()  # (N_rooms,)

        if batch_size < DEDUP_MIN_BATCH:
            x_hidden = self._embed_tokens(
                x_gs.tokens.x_column,
                x_gs.tokens.x_value,
                x_gs.tokens.effect_rows,
                x_gs.tokens.modifier_rows,
                x_entity_kind,
                x_room_row,
                x_hidden_map[x_map_room, x_node_room],
            )  # (N_tok, D)
        else:
            # Each unique token runs once and gathers back, up to a hash collision and rounding
            x_rep, x_is_rep, x_inverse = self._unique_tokens(
                x_gs.tokens, x_entity_kind, x_room_row, x_map_room, x_node_room
            )
            x_room_rep = torch.nonzero(x_token_kind[x_rep] == TokenKind.ROOMS).squeeze(1)
            x_room_row = x_rep[x_room_rep]
            x_node_hidden = x_hidden_map[
                x_map_of_state[x_state[x_room_row]], x_gs.tokens.x_node_idx[x_room_row].long()
            ]
            x_hidden = self._embed_tokens(
                x_gs.tokens.x_column[x_rep],
                x_gs.tokens.x_value[x_rep],
                _take_owned(x_gs.tokens.effect_rows, x_rep, x_is_rep),
                _take_owned(x_gs.tokens.modifier_rows, x_rep, x_is_rep),
                x_entity_kind[x_rep],
                x_room_rep,
                x_node_hidden,
            )
            x_hidden = x_hidden.index_select(0, x_inverse)  # (N_tok, D)

        # The pending halt owns no token: its Effect row is a global-context input instead
        x_effect_pending = x_gs.effect_pending
        x_pending_state = torch.repeat_interleave(
            x_effect_pending.x_count, output_size=x_effect_pending.x_column.shape[0]
        )  # (P,)
        x_pending_valid = _project_sparse(
            self._effects_mlp, x_effect_pending.x_column, x_effect_pending.x_value
        )  # (P, D)
        x_pending = torch.zeros(
            batch_size,
            self._dim_entity,
            dtype=x_pending_valid.dtype,
            device=x_pending_valid.device,
        )
        x_pending = x_pending.index_copy(0, x_pending_state, x_pending_valid)  # (B, D)

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

    def _embed_tokens(
        self,
        x_column: torch.Tensor,
        x_value: torch.Tensor,
        x_effect_rows: X_OwnedSparse,
        x_modifier_rows: X_OwnedSparse,
        x_entity_kind: torch.Tensor,
        x_room_row: torch.Tensor,
        x_node_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Embed the tokens: all of what the projector does per token.

        Parameters
        ----------
        x_column : torch.Tensor
            (N, K) int16. The tokens' entries' columns.
        x_value : torch.Tensor
            (N, K) float32. The tokens' entries' values.
        x_effect_rows : X_OwnedSparse
            The Effect rows the tokens own.
        x_modifier_rows : X_OwnedSparse
            The Modifier rows the tokens own.
        x_entity_kind : torch.Tensor
            (N,) long. Each token's Entity kind.
        x_room_row : torch.Tensor
            (N_rooms,) long. The Room tokens' rows.
        x_node_hidden : torch.Tensor
            (N_rooms, dim_map). Each Room token's node features.

        Returns
        -------
        torch.Tensor
            (N, D). Each token's embedding.
        """
        # First layer: the shared columns' projection, then each token's Entity kind bias
        x_first = _project_sparse(self._pre_token, x_column, x_value) + SpreadEmbedding.apply(
            self._bias_entity_kind, x_entity_kind
        )

        # Rooms: each Room token's node in its state's Map, through _pre_room in place of the first
        # layer (no entries, no Entity kind bias)
        x_first[x_room_row] = self._pre_room(x_node_hidden)

        # Owned rows add into their owners, scaled by 1/sqrt(rows per owner), before the shared
        # ReLU/LayerNorm
        _add_owned(x_first, x_effect_rows, self._effects_mlp)
        _add_owned(x_first, x_modifier_rows, self._modifiers_mlp)

        # Shared second layer + norm on the rows
        return self._layer_norm(self._post_shared(F.relu(x_first)))

    def _unique_tokens(
        self,
        x_tokens: X_Tokens,
        x_entity_kind: torch.Tensor,
        x_room_row: torch.Tensor,
        x_map_room: torch.Tensor,
        x_node_room: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Find the unique tokens, by a 64-bit hash of everything `_embed_tokens` reads per token.

        Parameters
        ----------
        x_tokens : X_Tokens
            The tokens, N of them.
        x_entity_kind : torch.Tensor
            (N,) long. Each token's Entity kind.
        x_room_row : torch.Tensor
            (N_rooms,) long. The Room tokens' rows.
        x_map_room : torch.Tensor
            (N_rooms,) long. Each Room token's Map, among those the Map GNN ran on.
        x_node_room : torch.Tensor
            (N_rooms,) long. Each Room token's node in its Map.

        Returns
        -------
        x_rep : torch.Tensor
            (U,) long. One row per unique token among the N, in rising order.
        x_is_rep : torch.Tensor
            (N,) bool. Whether each row is in `x_rep`.
        x_inverse : torch.Tensor
            (N,) long. Each token's unique token, its index in `x_rep`.
        """
        num_tokens = x_entity_kind.shape[0]
        device = x_entity_kind.device
        x_weight_entry = self._hash_weights_entry
        (
            x_weight_entity_kind,
            x_weight_effects,
            x_weight_modifiers,
            x_weight_map,
            x_weight_node,
            x_weight_mix,
        ) = self._hash_weights_field.unbind()
        x_hash_effects = _hash_owned(x_tokens.effect_rows, x_weight_entry, x_weight_mix)
        x_hash_modifiers = _hash_owned(x_tokens.modifier_rows, x_weight_entry, x_weight_mix)
        x_hash = (
            _hash_rows(x_tokens, x_weight_entry)
            + x_entity_kind * x_weight_entity_kind
            + x_hash_effects * x_weight_effects
            + x_hash_modifiers * x_weight_modifiers
        )  # (N,)
        x_hash[x_room_row] += x_map_room * x_weight_map + x_node_room * x_weight_node
        x_uniq, x_inverse = torch.unique(x_hash, return_inverse=True)  # (U,), (N,)

        # One token per unique hash, in rising order so the owned rows stay in their owners' order
        x_any = torch.zeros(x_uniq.shape[0], dtype=torch.long, device=device)
        x_any[x_inverse] = torch.arange(num_tokens, device=device)
        x_is_rep = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        x_is_rep[x_any] = True
        x_rep = torch.nonzero_static(x_is_rep, size=x_uniq.shape[0]).squeeze(1)
        return x_rep, x_is_rep, (x_is_rep.cumsum(0) - 1)[x_any[x_inverse]]


def _owners(x_owned_sparse: X_OwnedSparse) -> torch.Tensor:
    """Find the token that owns each row.

    Parameters
    ----------
    x_owned_sparse : X_OwnedSparse
        The owned row kind's rows, M of them.

    Returns
    -------
    torch.Tensor
        (M,) long. Each row's token.
    """
    # Each owned row's owner, from the counts widened
    return torch.repeat_interleave(
        x_owned_sparse.x_count.long(), output_size=x_owned_sparse.x_column.shape[0]
    )


def _take_owned(
    x_owned_sparse: X_OwnedSparse, x_rep: torch.Tensor, x_is_rep: torch.Tensor
) -> X_OwnedSparse:
    """Keep only the rows the unique tokens own.

    Parameters
    ----------
    x_owned_sparse : X_OwnedSparse
        The owned row kind's rows, for all N tokens.
    x_rep : torch.Tensor
        (U,) long. The unique tokens' rows among the N, in rising order.
    x_is_rep : torch.Tensor
        (N,) bool. Whether each row is in `x_rep`.

    Returns
    -------
    X_OwnedSparse
        The rows of the U tokens, in their order.
    """
    x_row = torch.nonzero(x_is_rep[_owners(x_owned_sparse)]).squeeze(1)
    return X_OwnedSparse(
        x_column=x_owned_sparse.x_column[x_row],
        x_value=x_owned_sparse.x_value[x_row],
        x_count=x_owned_sparse.x_count[x_rep],
    )


def _hash_rows(x_block: X_Tokens | X_OwnedSparse, x_weight_entry: torch.Tensor) -> torch.Tensor:
    """Hash each of a block's rows: its entries' columns and their values' bits.

    Parameters
    ----------
    x_block : X_Tokens | X_OwnedSparse
        The block, its rows held sparse: (N, K) entries.
    x_weight_entry : torch.Tensor
        (2, _HASH_MAX_ENTRIES) long. The weights of a row's entries: columns, then values.

    Returns
    -------
    torch.Tensor
        (N,) long. Each row's hash.
    """
    num_entries = x_block.x_column.shape[1]
    x_bits = x_block.x_value.view(torch.int32).long()
    return ((x_block.x_column.long() + 1) * x_weight_entry[0, :num_entries]).sum(1) + (
        x_bits * x_weight_entry[1, :num_entries]
    ).sum(1)


def _hash_owned(
    x_owned_sparse: X_OwnedSparse, x_weight_entry: torch.Tensor, x_weight_mix: torch.Tensor
) -> torch.Tensor:
    """Hash each token's owned rows: the sum of their mixed hashes, as `_add_owned` sums them.

    Parameters
    ----------
    x_owned_sparse : X_OwnedSparse
        The owned row kind's rows, held sparse.
    x_weight_entry : torch.Tensor
        (2, _HASH_MAX_ENTRIES) long. The weights of a row's entries: columns, then values.
    x_weight_mix : torch.Tensor
        () long. The mix's multiplier.

    Returns
    -------
    torch.Tensor
        (N,) long. Each token's hash, 0 if it owns no rows.
    """
    # Mix first: a sum of linear hashes would tie two Modifiers that swap their stacks
    x_hash_row = _hash_rows(x_owned_sparse, x_weight_entry)  # (M,)
    x_hash_row = (x_hash_row ^ (x_hash_row >> 32)) * x_weight_mix
    x_hash_row = x_hash_row ^ (x_hash_row >> 32)
    x_hash = torch.zeros(
        x_owned_sparse.x_count.shape[0], dtype=torch.long, device=x_hash_row.device
    )
    return x_hash.index_add_(0, _owners(x_owned_sparse), x_hash_row)


def _project_sparse(
    mapper: nn.Linear | nn.Sequential, x_column: torch.Tensor, x_value: torch.Tensor
) -> torch.Tensor:
    """Run `mapper` on the dense rows that the (N, K) entries stand for.

    Its first linear layer (the mapper, or a Sequential's first layer) sums its weight's columns
    at the entries' columns, scaled by their values, plus its bias; a Sequential's other layers
    follow.

    Parameters
    ----------
    mapper : nn.Linear | nn.Sequential
        A linear layer, or a Sequential whose first layer is linear.
    x_column : torch.Tensor
        (N, K) int16. The entries' columns.
    x_value : torch.Tensor
        (N, K) float32. The entries' values.

    Returns
    -------
    torch.Tensor
        (N, D). `mapper`'s output on each row.
    """
    linear = mapper if isinstance(mapper, nn.Linear) else mapper[0]
    x_rows = _SumSparseDenseBackward.apply(x_column, x_value, linear.weight)
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

    x_owner = _owners(x_owned_sparse)  # (M,)
    x_rows = _project_sparse(mapper, x_owned_sparse.x_column, x_owned_sparse.x_value)  # (M, D)
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
