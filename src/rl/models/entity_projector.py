import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.batches import get_row_segment
from src.rl.batches import get_segment_first_row
from src.rl.encoding import COL_START_CHARACTER
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
from src.rl.types import EntityKind
from src.rl.types import Jagged
from src.rl.types import Sparse
from src.rl.types import StateBatch
from src.rl.types import TokenKind


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
                    COL_START_HEALTH_BLOCK,
                    COL_START_PRICE,
                    COL_START_STASIS_SLOT,
                    COL_START_TAG,
                ),
                dim=1,
            ):
                nn.init.orthogonal_(x_weight_range, gain=math.sqrt(2))

    def forward(
        self, state_batch: StateBatch, x_room_hidden: torch.Tensor
    ) -> tuple[Jagged[torch.Tensor], torch.Tensor]:
        """Embed `state_batch`'s window.

        Parameters
        ----------
        state_batch : StateBatch
            The game states.
        x_room_hidden : torch.Tensor
            (N_rooms, dim_map). Each Room token's node features from the Map GNN, in the Room
            tokens' order (see `MapGNN.forward`).

        Returns
        -------
        jag_projected : Jagged[torch.Tensor]
            The (N_tok, D) token embeddings, grouped by state in window order.
        x_effect_pending : torch.Tensor
            (B, D). Each state's embedded pending halt, zero where none is pending.
        """
        # Shapes: B = batch, D = embedding width, N_tok = number of tokens
        tokens = state_batch.tokens.rows

        # First layer: the shared columns' projection, then each token's Entity kind bias
        x_first = _project_sparse(self._pre_token, tokens.features) + SpreadEmbedding.apply(
            self._x_bias_entity_kind, tokens.x_entity_kind.long()
        )

        # Rooms: each Room token's node features, through _pre_room in place of the first layer
        num_rooms = x_room_hidden.shape[0]
        x_room_row = torch.nonzero_static(tokens.x_token_kind == TokenKind.ROOMS, size=num_rooms)
        x_room_row = x_room_row.squeeze(1)  # (N_rooms,)
        x_first[x_room_row] = self._pre_room(x_room_hidden)

        # Project owned rows
        x_effects = _project_sparse(self._effects_mlp, tokens.effects.rows)
        x_modifiers = _project_sparse(self._modifiers_mlp, tokens.modifiers.rows)

        # Add owned rows to their owners, scaled by 1/sqrt(rows_per_owner)
        _add_owned(x_first, x_effects, tokens.effects.x_lengths)
        _add_owned(x_first, x_modifiers, tokens.modifiers.x_lengths)

        # The pending halt owns no token: its Effect row is a global-context input instead
        effect_pending = state_batch.effect_pending
        x_effect_pending = _project_sparse(self._effects_mlp, effect_pending)  # (B, D)
        x_effect_pending = x_effect_pending * (effect_pending.x_lengths > 0).unsqueeze(1)

        # Shared second layer + norm on the rows
        x_hidden = self._layer_norm(self._post_shared(F.relu(x_first)))  # (N_tok, D)

        return Jagged(x_lengths=state_batch.tokens.x_lengths, rows=x_hidden), x_effect_pending


def _project_sparse(mapper: nn.Linear | nn.Sequential, block: Sparse) -> torch.Tensor:
    """Run `mapper` on the dense rows represented by `block`'s jagged entries.

    Its first linear layer (the mapper, or a Sequential's first layer) sums its weight's columns
    at the entries' columns, scaled by their values, plus its bias; a Sequential's other layers
    follow.

    Parameters
    ----------
    mapper : nn.Linear | nn.Sequential
        A linear layer, or a Sequential whose first layer is linear.
    block : Sparse
        The block, its rows held sparse.

    Returns
    -------
    torch.Tensor
        (N, D). `mapper`'s output on each row.
    """
    linear = mapper if isinstance(mapper, nn.Linear) else mapper[0]
    x_rows = _SumSparseDenseBackward.apply(
        block.x_lengths, block.rows.x_columns, block.rows.x_values, linear.weight
    )
    if linear.bias is not None:
        x_rows = x_rows + linear.bias
    if isinstance(mapper, nn.Sequential):
        for i in range(1, len(mapper)):
            x_rows = mapper[i](x_rows)
    return x_rows


def _add_owned(x_first: torch.Tensor, x_rows: torch.Tensor, x_count: torch.Tensor) -> None:
    """Add an owned row kind's rows into their owners' rows of `x_first`.

    Each token owns one segment of rows, and each row is scaled by 1/sqrt(its owner's count).

    Parameters
    ----------
    x_first : torch.Tensor
        (N_tok, D). The tokens' first-layer rows, updated in place.
    x_rows : torch.Tensor
        (M, D). The owned rows, projected, each token's after the previous token's.
    x_count : torch.Tensor
        (N_tok,) int16. Each token's number of owned rows.
    """
    # Each owned row's token
    x_row_token = get_row_segment(x_count, x_rows.shape[0])  # (M,)

    # Sum projected owned rows to their respective owners' first-layer output representations
    x_first.index_add_(
        0, x_row_token, x_rows / x_count[x_row_token].to(x_rows.dtype).sqrt().unsqueeze(1)
    )


class _SumSparseDenseBackward(torch.autograd.Function):
    """`F.embedding_bag` over the sparse entries, with the weight's gradient as one GEMM.

    The GEMM runs over the entries densified to (N, in) rows: `embedding_bag`'s own backward sorts
    the entries and sums each column's serially.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x_lengths: torch.Tensor,
        x_column: torch.Tensor,
        x_value: torch.Tensor,
        x_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Run N rows' flat sparse entries through a linear layer's weight, bias aside.

        Sums the weight's columns at the entries' columns, scaled by their values.

        Parameters
        ----------
        x_lengths : torch.Tensor
            (N,) int16. Each row's number of entries, including empty rows.
        x_column : torch.Tensor
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
        ctx.save_for_backward(x_lengths, x_column, x_value)
        ctx.num_columns = x_weight.shape[1]

        # Each row's first entry, then the end
        x_row_first_entry = F.pad(
            get_segment_first_row(x_lengths), (0, 1), value=x_column.shape[0]
        )  # (N + 1,)

        # The transposed weight copied whole: as a strided view, a GPU reads its rows uncoalesced
        return F.embedding_bag(
            x_column.int(),
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
            The row lengths take no gradient.
        None
            The entries' columns take no gradient.
        None
            The entries' values take no gradient.
        torch.Tensor
            (D, in). The weight's gradient.
        """
        x_lengths, x_column, x_value = ctx.saved_tensors

        # Each entry's row
        x_entry_row = get_row_segment(x_lengths, x_column.shape[0])  # (F,)
        x_dense = torch.zeros(
            x_lengths.shape[0], ctx.num_columns, dtype=x_grad.dtype, device=x_grad.device
        )
        x_dense.view(-1).scatter_add_(0, x_entry_row * ctx.num_columns + x_column.long(), x_value)
        return None, None, None, x_grad.t() @ x_dense  # (D, in)
