import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.batches import get_entity_kind_rows
from src.rl.batches import get_owner_first_row
from src.rl.batches import get_row_owner
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
from src.rl.encoding import ENCODING_DIM_TOKEN
from src.rl.models.spread_embedding import SpreadEmbedding
from src.rl.types import Dense
from src.rl.types import EntityKind
from src.rl.types import Live
from src.rl.types import Sparse
from src.rl.types import States
from src.rl.types import Tokens


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

        # First layer: the shared columns' projection, then each token's Entity kind bias
        x_first = _project_sparse(self._pre_token, features.sparse) + SpreadEmbedding.apply(
            self._x_bias_entity_kind, tokens.x_entity_kind.long()
        )

        # Rooms: each Room token's node features, through _pre_room in place of the first layer
        num_rooms = x_room_hidden.shape[0]
        x_room_row = get_entity_kind_rows(tokens.x_entity_kind, EntityKind.ROOM, num_rooms)
        x_first[x_room_row] = self._pre_room(x_room_hidden)

        # Project owned rows
        x_effect = _project_sparse(self._effects_mlp, features.effects.sparse)
        x_modifier = _project_sparse(self._modifiers_mlp, features.modifiers.sparse)

        # Add owned rows to their owners, scaled by 1/sqrt(rows_per_owner)
        _add_owned(x_first, x_effect, features.effects.x_count)
        _add_owned(x_first, x_modifier, features.modifiers.x_count)

        # The pending halt owns no token: its Effect row is a global-context input instead
        effect_pending = states.effect_pending
        x_effect_pending = _project_sparse(self._effects_mlp, effect_pending)  # (B, D)
        x_effect_pending = x_effect_pending * (effect_pending.x_count > 0).unsqueeze(1)

        # Shared second layer + norm on the rows
        x_hidden = self._layer_norm(self._post_shared(F.relu(x_first)))  # (N_tok, D)

        return Tokens(
            features=Dense(x_value=x_hidden),
            x_count=tokens.x_count,
            x_collection_kind=tokens.x_collection_kind,
            x_entity_kind=tokens.x_entity_kind,
            x_legal_selection=tokens.x_legal_selection,
            x_legal_target=tokens.x_legal_target,
            x_needs_target=tokens.x_needs_target,
        ), x_effect_pending


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
