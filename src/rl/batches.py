"""Operations on a batch of game states: allocate, read, join, copy, count and select."""

from collections.abc import Callable
from collections.abc import Iterator
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
from src.rl.types import Jagged
from src.rl.types import JaggedReserved
from src.rl.types import SparseEntries
from src.rl.types import SparseReserved
from src.rl.types import StateBatch
from src.rl.types import StateBatchReserved
from src.rl.types import StateCounts
from src.rl.types import States
from src.rl.types import TokenKind
from src.rl.types import Tokens


__all__ = [
    "allocate_states",
    "cat_states",
    "iter_tensors",
    "map_tensors",
    "read_states",
    "slice_state_batch",
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


def allocate_states(num_buffers: int, buffer_size: int) -> StateBatchReserved:
    """Allocate reusable CPU capacity for every buffer's states.

    Parameters
    ----------
    num_buffers : int
        The number of independent buffers.
    buffer_size : int
        The number of states written into each buffer.

    Returns
    -------
    StateBatchReserved
        Ordinary CPU tensors with a leading buffer axis, sized for the maximum token, row and
        feature-entry counts, with every length zero.
    """
    # Maximum number of tokens per buffer
    num_tokens = buffer_size * MAX_TOKENS

    # Reserve memory for the rows the states own
    tokens = JaggedReserved(
        x_lengths=torch.zeros(num_buffers, buffer_size, dtype=torch.long),
        rows=Tokens(
            features=_allocate_sparse(num_buffers, num_tokens, ENCODING_DIM_SPARSE_TOKEN),
            x_token_kind=torch.zeros(num_buffers, num_tokens, dtype=torch.int16),
            x_entity_kind=torch.zeros(num_buffers, num_tokens, dtype=torch.int16),
            x_legal_selection=torch.zeros(
                num_buffers, num_tokens, NUM_ACTION_TYPES, dtype=torch.bool
            ),
            x_legal_target=torch.zeros(
                num_buffers, num_tokens, NUM_ACTION_TYPES, dtype=torch.bool
            ),
            x_needs_target=torch.zeros(num_buffers, num_tokens, dtype=torch.bool),
            effects=JaggedReserved(
                x_lengths=torch.zeros(num_buffers, num_tokens, dtype=torch.int16),
                rows=_allocate_sparse(
                    num_buffers, buffer_size * MAX_EFFECTS, ENCODING_DIM_SPARSE_EFFECT
                ),
            ),
            modifiers=JaggedReserved(
                x_lengths=torch.zeros(num_buffers, num_tokens, dtype=torch.int16),
                rows=_allocate_sparse(
                    num_buffers, buffer_size * MAX_MODIFIERS, ENCODING_DIM_SPARSE_MODIFIER
                ),
            ),
        ),
    )
    effect_pending = _allocate_sparse(num_buffers, buffer_size, ENCODING_DIM_SPARSE_EFFECT)

    # One count column per Jagged among those rows
    num_jaggeds = _count_jaggeds(tokens) + _count_jaggeds(effect_pending)

    # Reserve memory for states
    return States(
        tokens=tokens,
        effect_pending=effect_pending,
        x_context=torch.zeros(num_buffers, buffer_size, ENCODING_DIM_CONTEXT, dtype=torch.float32),
        x_map_grid=torch.zeros(
            num_buffers, buffer_size, MAP_HEIGHT, MAP_WIDTH, NUM_CHANNELS, dtype=torch.uint8
        ),
        x_legal_action_type=torch.zeros(
            num_buffers, buffer_size, NUM_ACTION_TYPES, dtype=torch.bool
        ),
        state_counts=StateCounts(
            x_token_kind_count=torch.zeros(
                num_buffers, buffer_size, len(TokenKind), dtype=torch.int16
            ),
            x_jagged_count=torch.zeros(num_buffers, buffer_size, num_jaggeds, dtype=torch.int32),
            x_draw_order_visible=torch.zeros(num_buffers, buffer_size, dtype=torch.bool),
        ),
    )


def _count_jaggeds(value: Any) -> int:
    """Count the Jaggeds under `value`, itself included.

    Parameters
    ----------
    value : Any
        A tensor, a reserved table or a nested dataclass record of them.

    Returns
    -------
    int
        The number of Jaggeds.
    """
    if isinstance(value, torch.Tensor):
        return 0

    if isinstance(value, JaggedReserved):
        return 1 + _count_jaggeds(value.rows)

    return sum(_count_jaggeds(getattr(value, field.name)) for field in fields(value))


def _allocate_sparse(num_buffers: int, num_rows: int, entries_per_row: int) -> SparseReserved:
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
    SparseReserved
        Ordinary CPU arrays with a leading buffer axis and every length zero.
    """
    num_entries = num_rows * entries_per_row
    return JaggedReserved(
        x_lengths=torch.zeros(num_buffers, num_rows, dtype=torch.int16),
        rows=SparseEntries(
            x_columns=torch.zeros(num_buffers, num_entries, dtype=torch.int16),
            x_values=torch.zeros(num_buffers, num_entries, dtype=torch.float32),
        ),
    )


def read_states(states: StateBatchReserved) -> StateBatch:
    """Generate the Jagged records of a completed buffer's written rows, as views.

    Parameters
    ----------
    states : StateBatchReserved
        One buffer's reserved states, its writer finished.

    Returns
    -------
    StateBatch
        Views of the written rows, sharing the buffer's storage. Retained observations must be
        copied before the buffer is written again.
    """
    return _read_rows(states, len(states.x_context))


def _read_rows(value: Any, num_rows: int) -> Any:
    """View the first `num_rows` rows of every tensor under `value`.

    Each reserved table comes out as the Jagged of its written rows.

    Parameters
    ----------
    value : Any
        A tensor, a reserved table or a nested dataclass record of them.
    num_rows : int
        The number of written rows in `value`'s tensors.

    Returns
    -------
    Any
        Views of the written rows, preserving nesting.
    """
    if isinstance(value, torch.Tensor):
        return value[:num_rows]

    # A table has as many written rows as its written lengths add up to
    if isinstance(value, JaggedReserved):
        x_lengths = value.x_lengths[:num_rows]
        num_rows = int(x_lengths.sum())
        return Jagged(x_lengths, _read_rows(value.rows, num_rows))

    return type(value)(
        **{field.name: _read_rows(getattr(value, field.name), num_rows) for field in fields(value)}
    )


def cat_states(state_batches: list[StateBatch]) -> StateBatch:
    """Concatenate compact worker batches or rollout steps field by field.

    Parameters
    ----------
    state_batches : list[StateBatch]
        At least one batch, in output order, with matching field dtypes and trailing shapes.

    Returns
    -------
    StateBatch
        The concatenated states and their descendants, preserving each batch's row order.
    """
    if not state_batches:
        raise ValueError("`cat_states` requires at least one batch")
    
    return _cat_records(state_batches)


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


def to_device(state_batch: StateBatch, device: torch.device) -> StateBatch:
    """Make an independent snapshot, using one pinned transfer for CUDA.

    Every result tensor is a view of the copied byte buffer. On CPU, the staging
    buffer itself owns the snapshot; no result aliases a worker's reusable storage.

    Parameters
    ----------
    state_batch : StateBatch
        The source states to snapshot.
    device : torch.device
        The destination device.

    Returns
    -------
    StateBatch
        The states on the destination device, with their shapes and dtypes preserved and
        storage independent of the source state_batch.
    """
    # Get all leaf Tensors
    x_leaves = list(iter_tensors(state_batch))

    # Calculate total buffer size and start / end indexes of each leaf Tensor
    idx_starts = []
    idx_end = 0 
    for x_leaf in x_leaves:
        idx_starts.append(idx_end)
        idx_end += (x_leaf.numel() * x_leaf.element_size() + 7) // 8 * 8

    # Allocate buffer in CPU memory and fill it. Each element stores a byte
    x_buffer = torch.empty(idx_end, dtype=torch.uint8, pin_memory=device.type == "cuda")
    for x_leaf, idx_start in zip(x_leaves, idx_starts):
        _bytes_as(x_buffer, x_leaf, idx_start).copy_(x_leaf)

    # Send buffer to device 
    x_buffer = x_buffer.to(device, non_blocking=True) if device.type == "cuda" else x_buffer

    # Define an iterator to get back each leaf Tensor from the buffer, and map it to `state_batch`
    iter_x_leaves = iter(_bytes_as(x_buffer, x_leaf, idx_start) for x_leaf, idx_start in zip(x_leaves, idx_starts))
    return map_tensors(state_batch, lambda _: next(iter_x_leaves))


def _bytes_as(x_buffer: torch.Tensor, x_like: torch.Tensor, idx_start: int) -> torch.Tensor:
    size = x_like.numel() * x_like.element_size()
    return x_buffer[idx_start : idx_start + size].view(x_like.dtype).view(x_like.shape)


def _select_jagged_rows(
    value: Any, selectors_remaining: Iterator[slice | torch.Tensor], selector: slice | torch.Tensor
) -> Any:
    """Index every tensor under `value` with its rows' selector, depth first in field order.

    A Jagged's lengths are indexed with `selector`, like the tensors beside it; its rows take
    the next selector from `selectors_remaining`.

    Parameters
    ----------
    value : Any
        A tensor, or a table or nested dataclass record of tensors.
    selectors_remaining : Iterator[slice | torch.Tensor]
        The remaining Jaggeds' row selectors, each a slice or an index tensor, in Jagged order.
    selector : slice | torch.Tensor
        The rows to select at `value`'s level.

    Returns
    -------
    Any
        The selected rows, preserving record types and nesting.
    """
    if isinstance(value, torch.Tensor):
        return value[selector]

    if isinstance(value, Jagged):
        return Jagged(
            value.x_lengths[selector],
            _select_jagged_rows(value.rows, selectors_remaining, next(selectors_remaining)),
        )

    return type(value)(
        **{
            field.name: _select_jagged_rows(
                getattr(value, field.name), selectors_remaining, selector
            )
            for field in fields(value)
        }
    )


def take_minibatch_states(
    state_batch: StateBatch, x_state_ids: torch.Tensor, state_counts_cpu: StateCounts
) -> StateBatch:
    """Gather states and all their descendants.

    Parameters
    ----------
    state_batch : StateBatch
        The source states.
    x_state_ids : torch.Tensor
        (num_states,) long. The states to gather, in output order.
    state_counts_cpu : StateCounts
        Those states' counts, on the CPU.

    Returns
    -------
    StateBatch
        The selected states and their descendants, preserving order and duplicate selections.
    """
    # The selected states' rows in every Jagged, totalled on the host
    counts_total = state_counts_cpu.x_jagged_count.sum(0).tolist()

    # Each state's rows in every Jagged, and where they start
    x_jagged_count = state_batch.state_counts.x_jagged_count.long()  # (B, number_of_Jaggeds)
    x_starts = x_jagged_count.cumsum(0) - x_jagged_count  # (B, number_of_Jaggeds)

    # Each Jagged's source rows, in output order
    jagged_rows = [
        _expand_ranges(
            x_starts[x_state_ids, idx_jagged],
            x_jagged_count[x_state_ids, idx_jagged],
            count_total,
        )
        for idx_jagged, count_total in enumerate(counts_total)
    ]
    return _select_jagged_rows(state_batch, iter(jagged_rows), x_state_ids)


def _expand_ranges(
    x_starts: torch.Tensor, x_counts: torch.Tensor, count_total: int
) -> torch.Tensor:
    """List the rows of consecutive ranges, one range after another.

    Range i covers x_counts[i] rows from x_starts[i]: starts [4, 0] with counts [2, 3] give
    [4, 5, 0, 1, 2]. It is a concatenation of `torch.arange(start, start + count)` without the
    loop.

    Parameters
    ----------
    x_starts : torch.Tensor
        (num_ranges,) long. Each range's first row.
    x_counts : torch.Tensor
        (num_ranges,) long. Each range's number of rows.
    count_total : int
        The ranges' total number of rows.

    Returns
    -------
    torch.Tensor
        (count_total,) long. The rows, range after range.
    """
    # Where each range lands in the output: back to back
    x_starts_output = x_counts.cumsum(0) - x_counts

    # Every output row moves by its range's shift: source row = output row + shift
    x_shifts = (x_starts - x_starts_output)

    # Repeat the shift for each row in each state
    x_shifts = x_shifts.repeat_interleave(x_counts, output_size=count_total)

    # Sum each row's indexes in the minibatch to the shifts to get their source-batch indexes
    return x_shifts + torch.arange(count_total, device=x_starts.device)


def slice_state_batch(
    state_batch: StateBatch, start: int, stop: int, state_counts_cpu: StateCounts
) -> StateBatch:
    """View consecutive states and their descendants using CPU counts.

    Parameters
    ----------
    state_batch : StateBatch
        The source states.
    start : int
        The first state's index, inclusive.
    stop : int
        The last state's index, exclusive; at most the number of states.
    state_counts_cpu : StateCounts
        The source states' counts, on the CPU.

    Returns
    -------
    StateBatch
        Views of the selected states and descendants, sharing the source tensors' storage.
    """
    # Each Jagged's rows before the first state, and before the stop
    x_jagged_count_cpu = state_counts_cpu.x_jagged_count
    first = x_jagged_count_cpu[:start].sum(0).tolist()
    last = x_jagged_count_cpu[:stop].sum(0).tolist()
    return _select_jagged_rows(state_batch, map(slice, first, last), slice(start, stop))
