"""Operations on a batch of game states: allocate, read, join, copy, count and select."""

from collections.abc import Callable
from collections.abc import Iterator
from dataclasses import fields
from typing import Any, cast

import torch
import torch.nn.functional as F

from src.rl.constants import MAP_HEIGHT
from src.rl.constants import MAP_WIDTH
from src.rl.constants import MAX_EFFECTS
from src.rl.constants import MAX_MODIFIERS
from src.rl.constants import MAX_TOKENS
from src.rl.constants import NUM_ACTION_TYPES
from src.rl.schema.context import ENCODING_DIM_CONTEXT
from src.rl.schema.effect import ENCODING_DIM_SPARSE_EFFECT
from src.rl.schema.map_ import NUM_CHANNELS
from src.rl.schema.modifier import ENCODING_DIM_SPARSE_MODIFIER
from src.rl.schema.state import ENCODING_DIM_SPARSE_TOKEN
from src.rl.types import Jagged
from src.rl.types import JaggedReserved
from src.rl.types import MinibatchPlan
from src.rl.types import SparseEntries
from src.rl.types import SparseReserved
from src.rl.types import StateBatch
from src.rl.types import StateBatchLayout
from src.rl.types import StateBatchReserved
from src.rl.types import States
from src.rl.types import Tokens


__all__ = [
    "allocate_states",
    "build_state_batch_layout",
    "cat_states",
    "count_jagged_rows",
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

    # Reserve memory for states
    return States(
        tokens=JaggedReserved(
            x_lengths=torch.zeros(num_buffers, buffer_size, dtype=torch.long),
            rows=Tokens(
                features=_allocate_sparse(num_buffers, num_tokens, ENCODING_DIM_SPARSE_TOKEN),
                x_token_kind=torch.zeros(num_buffers, num_tokens, dtype=torch.int16),
                x_legal_selection=torch.zeros(
                    num_buffers, num_tokens, NUM_ACTION_TYPES, dtype=torch.bool
                ),
                x_legal_target=torch.zeros(
                    num_buffers, num_tokens, NUM_ACTION_TYPES, dtype=torch.bool
                ),
                x_needs_target=torch.zeros(num_buffers, num_tokens, dtype=torch.bool),
                x_node_idx=torch.zeros(num_buffers, num_tokens, dtype=torch.int16),
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
        ),
        effect_pending=_allocate_sparse(num_buffers, buffer_size, ENCODING_DIM_SPARSE_EFFECT),
        x_context=torch.zeros(num_buffers, buffer_size, ENCODING_DIM_CONTEXT, dtype=torch.float32),
        x_map_grid=torch.zeros(
            num_buffers, buffer_size, MAP_HEIGHT, MAP_WIDTH, NUM_CHANNELS, dtype=torch.uint8
        ),
        x_legal_action_type=torch.zeros(
            num_buffers, buffer_size, NUM_ACTION_TYPES, dtype=torch.bool
        ),
    )


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


def count_jagged_rows(state_batch: StateBatch) -> torch.Tensor:
    """Count the rows each state owns in every Jagged.

    Parameters
    ----------
    state_batch : StateBatch
        The states whose rows to count.

    Returns
    -------
    torch.Tensor
        (B, number_of_Jaggeds) long on the batch's device, in Jagged order.
    """
    x_token_lengths = state_batch.tokens.x_lengths
    x_off_state = torch.arange(len(x_token_lengths) + 1, device=x_token_lengths.device)
    return torch.stack(list(_iter_jagged_row_counts(state_batch, x_off_state)), dim=1)


def _iter_jagged_row_counts(value: Any, x_off_state: torch.Tensor) -> Iterator[torch.Tensor]:
    """Yield, depth first in field order, each Jagged's per-state row count under `value`.

    Parameters
    ----------
    value : Any
        A tensor, a table or a nested dataclass record of them.
    x_off_state : torch.Tensor
        (B + 1,) long. Where each state's rows start at `value`'s level.

    Yields
    ------
    torch.Tensor
        (B,) long. A Jagged's rows per state.
    """
    if isinstance(value, torch.Tensor):
        return

    if isinstance(value, Jagged):
        # Calculate the start index of each row
        x_off = F.pad(value.x_lengths.cumsum(0, dtype=torch.long), (1, 0))

        # Take the previous level's state-start index; gives each state's start index at this level
        x_off = x_off[x_off_state]

        # Number of rows each state owns in this Jagged
        yield x_off.diff()

        # Continue the recursion
        yield from _iter_jagged_row_counts(value.rows, x_off)

    else:
        for field in fields(value):
            yield from _iter_jagged_row_counts(getattr(value, field.name), x_off_state)


def build_state_batch_layout(x_counts_cpu: torch.Tensor, device: torch.device) -> StateBatchLayout:
    """Cache offsets once for an immutable batch, on the device and CPU as needed.

    Parameters
    ----------
    x_counts_cpu : torch.Tensor
        (B, number_of_Jaggeds) on the CPU. Each state's row and entry counts, in Jagged order.
    device : torch.device
        The batch's device.

    Returns
    -------
    StateBatchLayout
        The source offsets, valid while the batch's lengths remain unchanged.
    """
    # The `F.pad` call puts an all-zeros row at the top
    x_off_state_cpu = F.pad(x_counts_cpu.cumsum(0, dtype=torch.long), (0, 0, 1, 0))
    x_off_state = (
        x_off_state_cpu.pin_memory() if device.type == "cuda" else x_off_state_cpu
    ).to(device, non_blocking=True)

    return StateBatchLayout(x_off_state, x_off_state_cpu)


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
    state_batch: StateBatch, minibatch_plan: MinibatchPlan, state_batch_layout: StateBatchLayout
) -> StateBatch:
    """Gather planned states and all their descendants using cached source offsets.

    Parameters
    ----------
    state_batch : StateBatch
        The source states.
    minibatch_plan : MinibatchPlan
        The state indices in output order and the selected states' total row and entry counts.
    state_batch_layout : StateBatchLayout
        The source batch's cached offsets.

    Returns
    -------
    StateBatch
        The selected states and their descendants, preserving order and duplicate selections.
    """
    x_state_ids = minibatch_plan.x_state_ids

    x_starts = state_batch_layout.x_off_state[x_state_ids]  # (S, number_of_Jaggeds)
    x_counts = state_batch_layout.x_off_state[x_state_ids + 1] - x_starts  # (S, number_of_Jaggeds)

    # Each Jagged's source rows, in output order
    jagged_rows = [
        _expand_ranges(x_starts[:, idx_jagged], x_counts[:, idx_jagged], count_total)
        for idx_jagged, count_total in enumerate(minibatch_plan.counts_total)
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
    state_batch: StateBatch, start: int, stop: int, state_batch_layout: StateBatchLayout
) -> StateBatch:
    """View consecutive states and their descendants using CPU offsets.

    Parameters
    ----------
    state_batch : StateBatch
        The source states.
    start : int
        The first state's index, inclusive.
    stop : int
        The last state's index, exclusive; at most the number of states.
    state_batch_layout : StateBatchLayout
        The source StateBatch's cached offsets.

    Returns
    -------
    StateBatch
        Views of the selected states and descendants, sharing the source tensors' storage.
    """
    first, last = state_batch_layout.x_off_state_cpu[[start, stop]].tolist()
    return _select_jagged_rows(state_batch, map(slice, first, last), slice(start, stop))
