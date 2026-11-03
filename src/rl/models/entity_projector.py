import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.batches import get_owner_first_row
from src.rl.batches import get_row_owner
from src.rl.constants import DEDUP_MIN_BATCH
from src.rl.encoding import COL_START_CHARACTER
from src.rl.encoding import COL_START_EVENT_OPTION
from src.rl.encoding import COL_START_HEALTH_BLOCK
from src.rl.encoding import COL_START_MONSTER
from src.rl.encoding import COL_START_POTION
from src.rl.encoding import COL_START_PRICE
from src.rl.encoding import COL_START_RELIC
from src.rl.encoding import COL_START_STASIS_SLOT
from src.rl.encoding import COL_START_TAG
from src.rl.encoding import ENCODING_DIM_EFFECT
from src.rl.encoding import ENCODING_DIM_MODIFIER
from src.rl.encoding import ENCODING_DIM_SPARSE_EFFECT
from src.rl.encoding import ENCODING_DIM_SPARSE_MODIFIER
from src.rl.encoding import ENCODING_DIM_SPARSE_TOKEN
from src.rl.encoding import ENCODING_DIM_TOKEN
from src.rl.models.spread_embedding import SpreadEmbedding
from src.rl.types import Dense
from src.rl.types import EncodedFeatures
from src.rl.types import EntityKind
from src.rl.types import Live
from src.rl.types import OwnedRows
from src.rl.types import Sparse
from src.rl.types import States
from src.rl.types import TokenKind
from src.rl.types import Tokens


# The most entries in a row of any block the hash reads
_MAX_ENTRIES_HASH = max(
    ENCODING_DIM_SPARSE_TOKEN, ENCODING_DIM_SPARSE_EFFECT, ENCODING_DIM_SPARSE_MODIFIER
)


class EntityProjector(nn.Module):
    """Factored (AlphaStar-style) Entity embedder.

    One first layer over the shared columns plus each token's Entity kind bias, the Rooms' Map
    nodes through their own, the owned row kinds (Effects, Modifiers) added into their owners,
    then one shared second layer + LayerNorm. Returns the token embeddings grouped by state and
    the embedded pending halt.

    Parameters
    ----------
    dim_entity : int
        The width of every Entity token's embedding.
    dim_map : int
        The width of the Map GNN's node features.
    """

    _x_hash_weight_entry: torch.Tensor
    _x_hash_weight_field: torch.Tensor
    _x_hash_weight_room: torch.Tensor

    def __init__(self, dim_entity: int, dim_map: int) -> None:
        super().__init__()

        # Every Entity kind's first layer: one projection over the shared columns
        self._pre_token = nn.Linear(ENCODING_DIM_TOKEN, dim_entity, bias=False)

        # Per-Entity kind bias
        self._x_bias_entity_kind = nn.Parameter(torch.zeros(len(EntityKind), dim_entity))

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

        # The dedup hash's odd weights, from a generator of their own; excluded from checkpoints
        generator = torch.Generator().manual_seed(0)
        self.register_buffer(
            "_x_hash_weight_entry",
            torch.randint(1 << 62, (2, _MAX_ENTRIES_HASH), generator=generator) | 1,
            persistent=False,
        )
        self.register_buffer(
            "_x_hash_weight_field",
            torch.randint(1 << 62, (4,), generator=generator) | 1,
            persistent=False,
        )
        self.register_buffer(
            "_x_hash_weight_room",
            torch.randint(1 << 62, (dim_map,), generator=generator) | 1,
            persistent=False,
        )

    def init_token_weights(self) -> None:
        """Initialize each token feature range independently after the general linear init.

        Each range receives an orthogonal initialization with gain sqrt(2), preserving its
        scale regardless of the total number of token columns.
        """
        with torch.no_grad():
            for x_weight_range in torch.tensor_split(
                self._pre_token.weight,
                (
                    COL_START_RELIC,
                    COL_START_POTION,
                    COL_START_MONSTER,
                    COL_START_CHARACTER,
                    COL_START_EVENT_OPTION,
                    COL_START_HEALTH_BLOCK,
                    COL_START_PRICE,
                    COL_START_STASIS_SLOT,
                    COL_START_TAG,
                ),
                dim=1,
            ):
                nn.init.orthogonal_(x_weight_range, gain=math.sqrt(2))

    def forward(
        self, states: States[Live], x_room_hidden: torch.Tensor
    ) -> tuple[Tokens[Dense], torch.Tensor]:
        """Embed `states`' window.

        Parameters
        ----------
        states : States[Live]
            The game states.
        x_room_hidden : torch.Tensor
            (N_rooms, dim_map). Each Room token's node features from the Map GNN, in the Room
            tokens' order (see `MapGNN.forward`).

        Returns
        -------
        tokens_projected : Tokens[Dense]
            The (N_tok, D) token embeddings, grouped by state in window order, with their kinds
            and legal masks.
        x_effect_pending : torch.Tensor
            (B, D). Each state's embedded pending halt, zero where none is pending.
        """
        # Shapes: B = batch, D = embedding width, N_tok = number of tokens
        tokens = states.tokens
        features = tokens.features
        batch_size = tokens.x_count.shape[0]

        # Rooms: the Room tokens' rows, in the order of their node features
        num_rooms = x_room_hidden.shape[0]
        x_room_row = torch.nonzero_static(tokens.x_token_kind == TokenKind.ROOMS, size=num_rooms)
        x_room_row = x_room_row.squeeze(1)  # (N_rooms,)

        if batch_size < DEDUP_MIN_BATCH:
            x_hidden = self._embed_tokens(
                features, tokens.x_entity_kind, x_room_row, x_room_hidden
            )  # (N_tok, D)
        else:
            # Each unique token runs once and gathers back, up to a hash collision and rounding
            x_is_rep, x_inverse = self._unique_tokens(tokens, x_room_row, x_room_hidden)
            x_room_rep = torch.nonzero(tokens.x_token_kind[x_is_rep] == TokenKind.ROOMS)
            x_room_rep = x_room_rep.squeeze(1)
            x_hidden = self._embed_tokens(
                EncodedFeatures(
                    sparse=_take_sparse(features.sparse, x_is_rep),
                    effects=_take_owned(features.effects, x_is_rep),
                    modifiers=_take_owned(features.modifiers, x_is_rep),
                ),
                tokens.x_entity_kind[x_is_rep],
                x_room_rep,
                x_room_hidden[x_is_rep[x_room_row]],
            )
            x_hidden = x_hidden.index_select(0, x_inverse)  # (N_tok, D)

        # The pending halt owns no token: its Effect row is a global-context input instead
        effect_pending = states.effect_pending
        x_effect_pending = _project_sparse(self._effects_mlp, effect_pending)  # (B, D)
        x_effect_pending = x_effect_pending * (effect_pending.x_count > 0).unsqueeze(1)

        return Tokens(
            features=Dense(x_value=x_hidden),
            x_count=tokens.x_count,
            x_token_kind=tokens.x_token_kind,
            x_entity_kind=tokens.x_entity_kind,
            x_legal_selection=tokens.x_legal_selection,
            x_legal_target=tokens.x_legal_target,
            x_needs_target=tokens.x_needs_target,
        ), x_effect_pending

    def _embed_tokens(
        self,
        features: EncodedFeatures[Live],
        x_entity_kind: torch.Tensor,
        x_room_row: torch.Tensor,
        x_room_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Embed the tokens: all of what the projector does per token.

        Parameters
        ----------
        features : EncodedFeatures[Live]
            The tokens' sparse features, one row per token, and the Effect and Modifier rows
            they own.
        x_entity_kind : torch.Tensor
            (N,) int16. Each token's Entity kind.
        x_room_row : torch.Tensor
            (N_rooms,) long. The Room tokens' rows.
        x_room_hidden : torch.Tensor
            (N_rooms, dim_map). Each Room token's node features.

        Returns
        -------
        torch.Tensor
            (N, D). Each token's embedding.
        """
        # First layer: the shared columns' projection, then each token's Entity kind bias
        x_first = _project_sparse(self._pre_token, features.sparse) + SpreadEmbedding.apply(
            self._x_bias_entity_kind, x_entity_kind.long()
        )

        # Rooms: each Room token's node features, through _pre_room in place of the first layer
        x_first[x_room_row] = self._pre_room(x_room_hidden)

        # Project owned rows
        x_effect = _project_sparse(self._effects_mlp, features.effects.sparse)
        x_modifier = _project_sparse(self._modifiers_mlp, features.modifiers.sparse)

        # Add owned rows to their owners, scaled by 1/sqrt(rows_per_owner)
        _add_owned(x_first, x_effect, features.effects.x_count)
        _add_owned(x_first, x_modifier, features.modifiers.x_count)

        # Shared second layer + norm on the rows
        return self._layer_norm(self._post_shared(F.relu(x_first)))

    def _unique_tokens(
        self,
        tokens: Tokens[EncodedFeatures[Live]],
        x_room_row: torch.Tensor,
        x_room_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find the unique tokens, by a 64-bit hash of everything `_embed_tokens` reads per token.

        Parameters
        ----------
        tokens : Tokens[EncodedFeatures[Live]]
            The tokens, N of them.
        x_room_row : torch.Tensor
            (N_rooms,) long. The Room tokens' rows.
        x_room_hidden : torch.Tensor
            (N_rooms, dim_map). Each Room token's node features.

        Returns
        -------
        x_is_rep : torch.Tensor
            (N,) bool. Whether each token is the one row of its unique token that runs.
        x_inverse : torch.Tensor
            (N,) long. Each token's unique token, its index among the rows `x_is_rep` marks.
        """
        features = tokens.features
        num_tokens = tokens.x_entity_kind.shape[0]
        device = tokens.x_entity_kind.device
        x_weight_entry = self._x_hash_weight_entry
        (
            x_weight_entity_kind,
            x_weight_effects,
            x_weight_modifiers,
            x_weight_mix,
        ) = self._x_hash_weight_field.unbind()
        x_hash_effects = _hash_owned(features.effects, x_weight_entry, x_weight_mix)
        x_hash_modifiers = _hash_owned(features.modifiers, x_weight_entry, x_weight_mix)
        x_hash = (
            _hash_sparse(features.sparse, x_weight_entry)
            + tokens.x_entity_kind.long() * x_weight_entity_kind
            + x_hash_effects * x_weight_effects
            + x_hash_modifiers * x_weight_modifiers
        )  # (N,)

        # A Room token's node features, by their bits
        x_room_bit = x_room_hidden.detach().view(torch.int32).long()  # (N_rooms, dim_map)
        x_hash[x_room_row] += (x_room_bit * self._x_hash_weight_room).sum(1)
        x_uniq, x_inverse = torch.unique(x_hash, return_inverse=True)  # (U,), (N,)

        # One token per unique hash, in rising order so the owned rows stay in their owners' order
        x_any = torch.zeros(x_uniq.shape[0], dtype=torch.long, device=device)
        x_any[x_inverse] = torch.arange(num_tokens, device=device)
        x_is_rep = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        x_is_rep[x_any] = True
        return x_is_rep, (x_is_rep.cumsum(0) - 1)[x_any[x_inverse]]


def _take_sparse(sparse: Sparse[Live], x_keep: torch.Tensor) -> Sparse[Live]:
    """Keep the rows `x_keep` marks, with their entries, in their order.

    Parameters
    ----------
    sparse : Sparse[Live]
        The rows, R of them.
    x_keep : torch.Tensor
        (R,) bool. Whether to keep each row.

    Returns
    -------
    Sparse[Live]
        The kept rows.
    """
    # Each entry's row, and the entries of the kept rows
    x_entry_row = get_row_owner(sparse.x_count, sparse.x_col.shape[0])  # (F,)
    x_entry = torch.nonzero(x_keep[x_entry_row]).squeeze(1)
    return Sparse(
        x_col=sparse.x_col[x_entry],
        x_value=sparse.x_value[x_entry],
        x_count=sparse.x_count[x_keep],
    )


def _take_owned(owned_rows: OwnedRows[Live], x_is_rep: torch.Tensor) -> OwnedRows[Live]:
    """Keep only the rows the unique tokens own.

    Parameters
    ----------
    owned_rows : OwnedRows[Live]
        The owned row kind's rows, consecutive per token, for all N tokens.
    x_is_rep : torch.Tensor
        (N,) bool. Whether each token is a unique token's row.

    Returns
    -------
    OwnedRows[Live]
        The rows of the unique tokens, in their order.
    """
    # Each owned row's token
    x_row_token = get_row_owner(owned_rows.x_count, owned_rows.sparse.x_count.shape[0])  # (M,)
    return OwnedRows(
        sparse=_take_sparse(owned_rows.sparse, x_is_rep[x_row_token]),
        x_count=owned_rows.x_count[x_is_rep],
    )


def _hash_sparse(sparse: Sparse[Live], x_weight_entry: torch.Tensor) -> torch.Tensor:
    """Hash each sparse row: its entries' columns and their values' bits, by place in the row.

    Parameters
    ----------
    sparse : Sparse[Live]
        The rows, R of them.
    x_weight_entry : torch.Tensor
        (2, _MAX_ENTRIES_HASH) long. The weights of a row's entries: columns, then values.

    Returns
    -------
    torch.Tensor
        (R,) long. Each row's hash.
    """
    num_entries = sparse.x_col.shape[0]
    x_entry_row = get_row_owner(sparse.x_count, num_entries)  # (F,)

    # Each entry's place in its row
    x_entry_place = torch.arange(num_entries, device=x_entry_row.device)
    x_entry_place = x_entry_place - get_owner_first_row(sparse.x_count)[x_entry_row]

    x_bit = sparse.x_value.view(torch.int32).long()
    x_hash_entry = (sparse.x_col.long() + 1) * x_weight_entry[0, x_entry_place] + (
        x_bit * x_weight_entry[1, x_entry_place]
    )  # (F,)
    x_hash = torch.zeros(sparse.x_count.shape[0], dtype=torch.long, device=x_hash_entry.device)
    return x_hash.index_add_(0, x_entry_row, x_hash_entry)


def _hash_owned(
    owned_rows: OwnedRows[Live], x_weight_entry: torch.Tensor, x_weight_mix: torch.Tensor
) -> torch.Tensor:
    """Hash each token's owned rows: the sum of their mixed hashes, as `_add_owned` sums them.

    Parameters
    ----------
    owned_rows : OwnedRows[Live]
        The owned row kind's rows, consecutive per token.
    x_weight_entry : torch.Tensor
        (2, _MAX_ENTRIES_HASH) long. The weights of a row's entries: columns, then values.
    x_weight_mix : torch.Tensor
        () long. The mix's multiplier.

    Returns
    -------
    torch.Tensor
        (N,) long. Each token's hash, 0 if it owns no rows.
    """
    # Mix first: a sum of linear hashes would tie two Modifiers that swap their stacks
    x_hash_row = _hash_sparse(owned_rows.sparse, x_weight_entry)  # (M,)
    x_hash_row = (x_hash_row ^ (x_hash_row >> 32)) * x_weight_mix
    x_hash_row = x_hash_row ^ (x_hash_row >> 32)

    # Each owned row's token
    x_row_token = get_row_owner(owned_rows.x_count, x_hash_row.shape[0])  # (M,)
    x_hash = torch.zeros(owned_rows.x_count.shape[0], dtype=torch.long, device=x_hash_row.device)
    return x_hash.index_add_(0, x_row_token, x_hash_row)


def _project_sparse(mapper: nn.Linear | nn.Sequential, sparse: Sparse[Live]) -> torch.Tensor:
    """Run `mapper` on the dense rows represented by `sparse`'s sparse entries.

    Its first linear layer (the mapper, or a Sequential's first layer) sums its weight's columns
    at the entries' columns, scaled by their values, plus its bias; a Sequential's other layers
    follow.

    Parameters
    ----------
    mapper : nn.Linear | nn.Sequential
        A linear layer, or a Sequential whose first layer is linear.
    sparse : Sparse[Live]
        The block, its rows held sparse.

    Returns
    -------
    torch.Tensor
        (N, D). `mapper`'s output on each row.
    """
    linear = mapper if isinstance(mapper, nn.Linear) else mapper[0]
    x_row = _SumSparseDenseBackward.apply(
        sparse.x_count, sparse.x_col, sparse.x_value, linear.weight
    )
    if linear.bias is not None:
        x_row = x_row + linear.bias
    if isinstance(mapper, nn.Sequential):
        for idx_layer in range(1, len(mapper)):
            x_row = mapper[idx_layer](x_row)
    return x_row


def _add_owned(x_first: torch.Tensor, x_row: torch.Tensor, x_count: torch.Tensor) -> None:
    """Add an owned row kind's rows into their owners' rows of `x_first`.

    Each token owns consecutive rows, and each row is scaled by 1/sqrt(its owner's count).

    Parameters
    ----------
    x_first : torch.Tensor
        (N_tok, D). The tokens' first-layer rows, updated in place.
    x_row : torch.Tensor
        (M, D). The owned rows, projected, each token's after the previous token's.
    x_count : torch.Tensor
        (N_tok,) int16. Each token's number of owned rows.
    """
    # Each owned row's token
    x_row_token = get_row_owner(x_count, x_row.shape[0])  # (M,)

    # Sum projected owned rows to their respective owners' first-layer output representations
    x_first.index_add_(
        0, x_row_token, x_row / x_count[x_row_token].to(x_row.dtype).sqrt().unsqueeze(1)
    )


class _SumSparseDenseBackward(torch.autograd.Function):
    """`F.embedding_bag` over the sparse entries, with the weight's gradient as one GEMM.

    The GEMM runs over the entries densified to (N, in) rows: `embedding_bag`'s own backward sorts
    the entries and sums each column's serially.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x_count: torch.Tensor,
        x_col: torch.Tensor,
        x_value: torch.Tensor,
        x_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Run N rows' flat sparse entries through a linear layer's weight, bias aside.

        Sums the weight's columns at the entries' columns, scaled by their values.

        Parameters
        ----------
        x_count : torch.Tensor
            (N,) int16. Each row's number of entries, including empty rows.
        x_col : torch.Tensor
            (F,) int16. The entries' columns, row after row.
        x_value : torch.Tensor
            (F,) float32. The entries' values, in the same order.
        x_weight : torch.Tensor
            (D, in). The linear layer's weight.

        Returns
        -------
        torch.Tensor
            (N, D). Each row's sum.
        """
        ctx.save_for_backward(x_count, x_col, x_value)
        ctx.num_cols = x_weight.shape[1]

        # Each row's first entry, then the end
        x_row_first_entry = F.pad(
            get_owner_first_row(x_count), (0, 1), value=x_col.shape[0]
        )  # (N + 1,)

        # The transposed weight copied whole: as a strided view, a GPU reads its rows uncoalesced
        return F.embedding_bag(
            x_col.int(),
            x_weight.t().contiguous(),
            offsets=x_row_first_entry,
            per_sample_weights=x_value,
            mode="sum",
            include_last_offset=True,
        )

    @staticmethod
    def backward(ctx: Any, x_grad: torch.Tensor) -> tuple[None, None, None, torch.Tensor]:
        """Take the weight's gradient as one GEMM over the entries densified to (N, in) rows.

        Parameters
        ----------
        x_grad : torch.Tensor
            (N, D). Each row's output gradient.

        Returns
        -------
        None
            The row counts take no gradient.
        None
            The entries' columns take no gradient.
        None
            The entries' values take no gradient.
        torch.Tensor
            (D, in). The weight's gradient.
        """
        x_count, x_col, x_value = ctx.saved_tensors

        # Each entry's row
        x_entry_row = get_row_owner(x_count, x_col.shape[0])  # (F,)
        x_dense = torch.zeros(
            x_count.shape[0], ctx.num_cols, dtype=x_grad.dtype, device=x_grad.device
        )
        x_dense.view(-1).scatter_add_(
            0, x_entry_row * ctx.num_cols + x_col.long(), x_value
        )
        return None, None, None, x_grad.t() @ x_dense  # (D, in)
