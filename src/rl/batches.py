"""Operations on a batch of game states: allocate, read, join, copy and select."""

from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from dataclasses import fields
from typing import Any, cast

import torch

from src.rl.constants import MAP_HEIGHT
from src.rl.constants import MAP_WIDTH
from src.rl.constants import MAX_EFFECTS
from src.rl.constants import MAX_MODIFIERS
from src.rl.constants import MAX_TOKENS
from src.rl.constants import NUM_ACTION_TYPES
from src.rl.encoding import ENCODING_DIM_CONTEXT
from src.rl.encoding import ENCODING_DIM_SPARSE_EFFECT
from src.rl.encoding import ENCODING_DIM_SPARSE_MODIFIER
from src.rl.encoding import ENCODING_DIM_SPARSE_TOKEN
from src.rl.encoding import NUM_CHANNELS
from src.rl.types import EncodedFeatures
from src.rl.types import EntityKind
from src.rl.types import Live
from src.rl.types import OwnedRows
from src.rl.types import Reserved
from src.rl.types import Sparse
from src.rl.types import StateCollection
from src.rl.types import States
from src.rl.types import TokenKind
from src.rl.types import Tokens


__all__ = [
    "allocate_states",
    "cat_states",
    "get_owner_first_row",
    "get_row_owner",
    "iter_tensors",
    "map_tensors",
    "read_states",
    "slice_states",
    "take_minibatch_states",
    "to_device",
]


def iter_tensors(value: Any) -> Iterator[torch.Tensor]:
    """Walk the tensor leaves of the data records in declaration order.

    Parameters
    ----------
    value : Any
        A tensor or nested dataclass record whose leaves are tensors.

    Yields
    ------
    torch.Tensor
        Each tensor leaf, in field declaration order.
    """
    if isinstance(value, torch.Tensor):
        yield value
    else:
        for field in fields(value):
            yield from iter_tensors(getattr(value, field.name))


def map_tensors[T](value: T, fn: Callable[[torch.Tensor], torch.Tensor]) -> T:
    """Rebuild data records with a function applied to each tensor leaf.

    Parameters
    ----------
    value : T
        A tensor or nested dataclass record whose leaves are tensors.
    fn : Callable[[torch.Tensor], torch.Tensor]
        The function to apply to each tensor leaf.

    Returns
    -------
    T
        The transformed tensor or rebuilt record, preserving record types and nesting.
    """
    if isinstance(value, torch.Tensor):
        return cast(T, fn(value))

    record = cast(Any, value)
    return type(record)(
        **{field.name: map_tensors(getattr(record, field.name), fn) for field in fields(record)}
    )


def get_row_owner(x_count: torch.Tensor, num_rows: int) -> torch.Tensor:
    """Find each row's owner.

    The rows sit owner after owner: counts [3, 2, 4] give [0, 0, 0, 1, 1, 2, 2, 2, 2].

    Parameters
    ----------
    x_count : torch.Tensor
        (num_owners,) integer. Each owner's number of rows.
    num_rows : int
        The owners' total number of rows, as the host knows it: no size is read back from the
        device.

    Returns
    -------
    torch.Tensor
        (num_rows,) long. Each row's owner.
    """
    return torch.repeat_interleave(x_count.long(), output_size=num_rows)


def get_owner_first_row(x_count: torch.Tensor) -> torch.Tensor:
    """Find each owner's first row.

    The rows sit owner after owner: counts [3, 2, 4] give [0, 3, 5].

    Parameters
    ----------
    x_count : torch.Tensor
        (num_owners, ...) integer. Each owner's number of rows. Any further axes hold
        independent groupings, each totalled on its own.

    Returns
    -------
    torch.Tensor
        (num_owners, ...) long. Each owner's first row.
    """
    return x_count.cumsum(0) - x_count


def get_entity_kind_rows(
    x_entity_kind: torch.Tensor, entity_kind: EntityKind, num_rows: int
) -> torch.Tensor:
    """Find the rows of one Entity kind's tokens.

    The tokens sit state after state, so their rows come out state after state too.

    Parameters
    ----------
    x_entity_kind : torch.Tensor
        (N,) int16. Each token's Entity kind.
    entity_kind : EntityKind
        The Entity kind whose tokens to find.
    num_rows : int
        The number of tokens of that Entity kind, as the host knows it: no size is read back
        from the device.

    Returns
    -------
    torch.Tensor
        (num_rows,) long. Those tokens' rows, in increasing order.
    """
    return torch.nonzero_static(x_entity_kind == entity_kind, size=num_rows).squeeze(1)


def allocate_states(num_buffers: int, buffer_size: int) -> States[Reserved]:
    """Allocate reusable CPU capacity for every buffer's states.

    Parameters
    ----------
    num_buffers : int
        The number of independent buffers, positive.
    buffer_size : int
        The number of states written into each buffer, nonnegative.

    Returns
    -------
    States[Reserved]
        Ordinary CPU tensors with a leading buffer axis, sized for the maximum token, row and
        feature-entry counts, with every count zero. Select a buffer before writing or reading.
    """
    if num_buffers <= 0 or buffer_size < 0:
        raise ValueError("Buffer count must be positive and buffer size nonnegative")

    # Maximum number of tokens per buffer
    num_tokens = buffer_size * MAX_TOKENS

    # Reserve memory for the rows the states own
    tokens: Tokens[EncodedFeatures[Reserved]] = Tokens(
        features=EncodedFeatures(
            sparse=_allocate_sparse(num_buffers, num_tokens, ENCODING_DIM_SPARSE_TOKEN),
            effects=OwnedRows(
                sparse=_allocate_sparse(
                    num_buffers, buffer_size * MAX_EFFECTS, ENCODING_DIM_SPARSE_EFFECT
                ),
                x_count=torch.zeros(num_buffers, num_tokens, dtype=torch.int16),
            ),
            modifiers=OwnedRows(
                sparse=_allocate_sparse(
                    num_buffers, buffer_size * MAX_MODIFIERS, ENCODING_DIM_SPARSE_MODIFIER
                ),
                x_count=torch.zeros(num_buffers, num_tokens, dtype=torch.int16),
            ),
        ),
        x_count=torch.zeros(num_buffers, buffer_size, dtype=torch.long),
        x_token_kind=torch.zeros(num_buffers, num_tokens, dtype=torch.int16),
        x_entity_kind=torch.zeros(num_buffers, num_tokens, dtype=torch.int16),
        x_legal_selection=torch.zeros(
            num_buffers, num_tokens, NUM_ACTION_TYPES, dtype=torch.bool
        ),
        x_legal_target=torch.zeros(num_buffers, num_tokens, NUM_ACTION_TYPES, dtype=torch.bool),
        x_needs_target=torch.zeros(num_buffers, num_tokens, dtype=torch.bool),
    )
    effect_pending = _allocate_sparse(num_buffers, buffer_size, ENCODING_DIM_SPARSE_EFFECT)

    # Reserve memory for states
    return States(
        tokens=tokens,
        effect_pending=effect_pending,
        x_context=torch.zeros(
            num_buffers, buffer_size, ENCODING_DIM_CONTEXT, dtype=torch.float32
        ),
        x_map_grid=torch.zeros(
            num_buffers, buffer_size, MAP_HEIGHT, MAP_WIDTH, NUM_CHANNELS, dtype=torch.uint8
        ),
        x_legal_action_type=torch.zeros(
            num_buffers, buffer_size, NUM_ACTION_TYPES, dtype=torch.bool
        ),
        x_total=torch.zeros(
            num_buffers, buffer_size, len(TokenKind) + len(StateCollection), dtype=torch.int32
        ),
    )


def _allocate_sparse(
    num_buffers: int, num_rows: int, entries_per_row: int
) -> Sparse[Reserved]:
    """Allocate one sparse table's rows and entries for every buffer.

    Parameters
    ----------
    num_buffers : int
        The number of buffers sharing each array's leading axis.
    num_rows : int
        Each buffer's maximum number of sparse rows.
    entries_per_row : int
        The maximum number of nonzero entries in a row.

    Returns
    -------
    Sparse[Reserved]
        Reserved CPU arrays with a leading buffer axis and every count zero.
    """
    num_entries = num_rows * entries_per_row
    return Sparse(
        x_col=torch.zeros(num_buffers, num_entries, dtype=torch.int16),
        x_value=torch.zeros(num_buffers, num_entries, dtype=torch.float32),
        x_count=torch.zeros(num_buffers, num_rows, dtype=torch.int16),
    )


def read_states(states_reserved: States[Reserved]) -> States[Live]:
    """View the populated collections of one completed buffer.

    Parameters
    ----------
    states_reserved : States[Reserved]
        One buffer's reserved states, its writer finished. Counts in unused tails are ignored.

    Returns
    -------
    States[Live]
        Views of the written rows, sharing the buffer's storage. Retained observations must be
        copied before the buffer is written again.
    """
    if states_reserved.x_context.ndim != 2:
        raise ValueError("Select one buffer before reading states")

    # Every state of the buffer is written: its populated rows are the slice of all of them
    return slice_states(
        states_reserved, 0, len(states_reserved.x_context), states_reserved.x_total
    )


def cat_states(batch_states: list[States[Live]]) -> States[Live]:
    """Concatenate compact worker batches or rollout steps field by field.

    Parameters
    ----------
    batch_states : list[States[Live]]
        At least one batch, in output order, with matching field dtypes and trailing shapes.

    Returns
    -------
    States[Live]
        The concatenated states and their descendants, preserving each batch's row order.
    """
    if not batch_states:
        raise ValueError("`cat_states` requires at least one batch")
    
    return _cat_records(batch_states)


def _cat_records(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(values)
    
    return type(first)(
        **{
            field.name: _cat_records([getattr(v, field.name) for v in values])
            for field in fields(first)
        }
    )


def to_device(states: States[Live], device: torch.device) -> States[Live]:
    """Make an independent snapshot, using one pinned transfer for CUDA.

    Every result tensor is a view of the copied byte buffer. On CPU, the staging
    buffer itself owns the snapshot; no result aliases a worker's reusable storage.

    Parameters
    ----------
    states : States[Live]
        The source states to snapshot.
    device : torch.device
        The destination device.

    Returns
    -------
    States[Live]
        The states on the destination device, with their shapes and dtypes preserved and
        storage independent of the source states.
    """
    # Get all leaf Tensors
    x_leaves = list(iter_tensors(states))

    # Calculate total buffer size and start / end indexes of each leaf Tensor
    idx_starts = []
    idx_stop = 0 
    for x_leaf in x_leaves:
        idx_starts.append(idx_stop)
        idx_stop += (x_leaf.numel() * x_leaf.element_size() + 7) // 8 * 8

    # Allocate buffer in CPU memory and fill it. Each element stores a byte
    x_buffer = torch.empty(idx_stop, dtype=torch.uint8, pin_memory=device.type == "cuda")
    for x_leaf, idx_start in zip(x_leaves, idx_starts):
        _bytes_as(x_buffer, x_leaf, idx_start).copy_(x_leaf)

    # Send buffer to device 
    x_buffer = x_buffer.to(device, non_blocking=True) if device.type == "cuda" else x_buffer

    # Define an iterator to get back each leaf Tensor from the buffer, and map it to `states`
    iter_x_leaves = iter(_bytes_as(x_buffer, x_leaf, idx_start) for x_leaf, idx_start in zip(x_leaves, idx_starts))
    return map_tensors(states, lambda _: next(iter_x_leaves))


def _bytes_as(x_buffer: torch.Tensor, x_like: torch.Tensor, idx_start: int) -> torch.Tensor:
    size = x_like.numel() * x_like.element_size()
    return x_buffer[idx_start : idx_start + size].view(x_like.dtype).view(x_like.shape)


def _select_sparse[TStorage](
    sparse: Sparse[TStorage],
    selector_rows: slice | torch.Tensor,
    selector_entries: slice | torch.Tensor,
) -> Sparse[Live]:
    """Select sparse rows and their entries using their respective selectors.

    Parameters
    ----------
    sparse : Sparse[TStorage]
        The source feature rows and their entries, live or reserved.
    selector_rows : slice | torch.Tensor
        The logical rows, including empty rows.
    selector_entries : slice | torch.Tensor
        Those rows' entries, in the selected row order.

    Returns
    -------
    Sparse[Live]
        The selected row counts, columns and values.
    """
    return Sparse(
        x_col=sparse.x_col[selector_entries],
        x_value=sparse.x_value[selector_entries],
        x_count=sparse.x_count[selector_rows],
    )


def _select_owned_rows[TStorage](
    owned_rows: OwnedRows[TStorage],
    selector_tokens: slice | torch.Tensor,
    selector_rows: slice | torch.Tensor,
    selector_entries: slice | torch.Tensor,
) -> OwnedRows[Live]:
    """Select some tokens' owned rows and their feature entries.

    Parameters
    ----------
    owned_rows : OwnedRows[TStorage]
        The source tokens' ownership counts and sparse rows, live or reserved.
    selector_tokens : slice | torch.Tensor
        The owning tokens.
    selector_rows : slice | torch.Tensor
        Their owned rows, in selected token order.
    selector_entries : slice | torch.Tensor
        Those rows' feature entries, in selected row order.

    Returns
    -------
    OwnedRows[Live]
        The selected ownership counts and feature rows.
    """
    return OwnedRows(
        sparse=_select_sparse(owned_rows.sparse, selector_rows, selector_entries),
        x_count=owned_rows.x_count[selector_tokens],
    )


def _select_states[TStorage](
    states: States[TStorage],
    selector_states: slice | torch.Tensor,
    selectors: Mapping[StateCollection, slice | torch.Tensor],
) -> States[Live]:
    """Select states using a named selector for each of their collections.

    Parameters
    ----------
    states : States[TStorage]
        The source states, live or reserved.
    selector_states : slice | torch.Tensor
        The states, in output order.
    selectors : Mapping[StateCollection, slice | torch.Tensor]
        Each collection's rows or entries belonging to those states, in output order.

    Returns
    -------
    States[Live]
        The selected states and all their descendants.
    """
    tokens = states.tokens
    selector_tokens = selectors[StateCollection.TOKENS]

    return States(
        tokens=Tokens(
            features=EncodedFeatures(
                sparse=_select_sparse(
                    tokens.features.sparse, selector_tokens,
                    selectors[StateCollection.TOKEN_ENTRIES],
                ),
                effects=_select_owned_rows(
                    tokens.features.effects, selector_tokens,
                    selectors[StateCollection.EFFECTS],
                    selectors[StateCollection.EFFECT_ENTRIES],
                ),
                modifiers=_select_owned_rows(
                    tokens.features.modifiers, selector_tokens,
                    selectors[StateCollection.MODIFIERS],
                    selectors[StateCollection.MODIFIER_ENTRIES],
                ),
            ),
            x_count=tokens.x_count[selector_states],
            x_token_kind=tokens.x_token_kind[selector_tokens],
            x_entity_kind=tokens.x_entity_kind[selector_tokens],
            x_legal_selection=tokens.x_legal_selection[selector_tokens],
            x_legal_target=tokens.x_legal_target[selector_tokens],
            x_needs_target=tokens.x_needs_target[selector_tokens],
        ),
        effect_pending=_select_sparse(
            states.effect_pending, selector_states, selectors[StateCollection.PENDING_ENTRIES]
        ),
        x_context=states.x_context[selector_states],
        x_map_grid=states.x_map_grid[selector_states],
        x_legal_action_type=states.x_legal_action_type[selector_states],
        x_total=states.x_total[selector_states],
    )


def take_minibatch_states(
    states: States[Live], x_idx_state: torch.Tensor, x_total_cpu: torch.Tensor
) -> States[Live]:
    """Gather states and all their descendants.

    Parameters
    ----------
    states : States[Live]
        The source states.
    x_idx_state : torch.Tensor
        (batch_size,) long. The states to gather, in output order.
    x_total_cpu : torch.Tensor
        Those states' totals, on the CPU.

    Returns
    -------
    States[Live]
        The selected states and their descendants, preserving order and duplicate selections.
    """
    # The selected states' rows in every collection, totalled on the host
    num_rows = x_total_cpu.sum(0).tolist()

    # Each state's rows in every collection, and where they start
    x_total = states.x_total.long()
    x_state_first_row = get_owner_first_row(x_total)

    # Each collection's source rows, in output order
    selectors = {
        collection: _expand_ranges(
            x_state_first_row[x_idx_state, collection],
            x_total[x_idx_state, collection],
            num_rows[collection],
        )
        for collection in StateCollection
    }
    return _select_states(states, x_idx_state, selectors)


def _expand_ranges(
    x_start: torch.Tensor, x_count: torch.Tensor, num_rows: int
) -> torch.Tensor:
    """List the rows of consecutive ranges, one range after another.

    Range i covers x_count[i] rows from x_start[i]: starts [4, 0] with counts [2, 3] give
    [4, 5, 0, 1, 2]. It is a concatenation of `torch.arange(start, start + count)` without the
    loop.

    Parameters
    ----------
    x_start : torch.Tensor
        (num_ranges,) long. Each range's first row.
    x_count : torch.Tensor
        (num_ranges,) long. Each range's number of rows.
    num_rows : int
        The ranges' total number of rows.

    Returns
    -------
    torch.Tensor
        (num_rows,) long. The rows, range after range.
    """
    # Where each range lands in the output: back to back
    x_start_output = get_owner_first_row(x_count)

    # Every output row moves by its range's shift: source row = output row + shift
    x_shift = (x_start - x_start_output)

    # Repeat the shift for each row in each state
    x_shift = x_shift.repeat_interleave(x_count, output_size=num_rows)

    # Sum each row's indexes in the minibatch to the shifts to get their source-batch indexes
    return x_shift + torch.arange(num_rows, device=x_start.device)


def slice_states[TStorage](
    states: States[TStorage], idx_start: int, idx_stop: int, x_total_cpu: torch.Tensor
) -> States[Live]:
    """View consecutive states and their descendants using CPU totals.

    Parameters
    ----------
    states : States[TStorage]
        The source states, live or reserved.
    idx_start : int
        The first state's index, inclusive.
    idx_stop : int
        The last state's index, exclusive; at most the number of states.
    x_total_cpu : torch.Tensor
        The source states' totals, on the CPU.

    Returns
    -------
    States[Live]
        Views of the selected states and descendants, sharing the source tensors' storage.
    """
    # Each collection's rows before the first state, and before the stop
    row_start = x_total_cpu[:idx_start].sum(0).tolist()
    row_stop = x_total_cpu[:idx_stop].sum(0).tolist()
    selectors = {
        collection: slice(row_start[collection], row_stop[collection])
        for collection in StateCollection
    }
    return _select_states(states, slice(idx_start, idx_stop), selectors)
