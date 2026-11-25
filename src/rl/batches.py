"""Operations on a batch of game states: allocate, read, join, copy and select."""

from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from dataclasses import fields, replace
from typing import Any, cast

import numpy as np
import torch
from slai import MAP_HEIGHT
from slai import MAP_WIDTH

from src.rl.constants import MAX_EFFECTS
from src.rl.constants import MAX_MODIFIERS
from src.rl.constants import MAX_TOKENS
from src.rl.constants import NUM_ACTION_TYPES
from src.rl.encoding import ENCODING_DIM_CONTEXT
from src.rl.encoding import ENCODING_DIM_ROOM
from src.rl.encoding import ENCODING_DIM_SPARSE_EFFECT
from src.rl.encoding import ENCODING_DIM_SPARSE_MODIFIER
from src.rl.encoding import ENCODING_DIM_SPARSE_TOKEN
from src.rl.encoding import MAP_TOPOLOGY_CHANNELS
from src.rl.types import CollectionKind
from src.rl.types import EncodedFeatures
from src.rl.types import EntityKind
from src.rl.types import Live
from src.rl.types import OwnedRows
from src.rl.types import Reserved
from src.rl.types import Sparse
from src.rl.types import StateCollection
from src.rl.types import States
from src.rl.types import Tokens


__all__ = [
    "allocate_states",
    "cat_states",
    "dedup_rows",
    "expand_ranges",
    "gather_states",
    "get_owner_first_row",
    "get_plan_unique",
    "get_row_owner",
    "get_unique_rep_row",
    "iter_tensors",
    "map_tensors",
    "read_states",
    "select_owned_rows",
    "select_sparse",
    "slice_states",
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
    """Rebuild data records with a function applied to each distinct tensor leaf.

    Parameters
    ----------
    value : T
        A tensor or nested dataclass record whose leaves are tensors.
    fn : Callable[[torch.Tensor], torch.Tensor]
        The function to apply to each tensor leaf.

    Returns
    -------
    T
        The transformed tensor or rebuilt record, preserving record types, nesting and shared
        tensor references. Device placement follows `fn`; use `to_device` to transfer states
        while keeping their CPU totals on the host.
    """
    return _map_tensors(value, fn, {})


def _map_tensors[T](
    value: T,
    fn: Callable[[torch.Tensor], torch.Tensor],
    x_mapped: dict[int, torch.Tensor],
) -> T:
    """Map each tensor once, retaining aliases across the record's fields."""
    if isinstance(value, torch.Tensor):
        key = id(value)
        if key not in x_mapped:
            x_mapped[key] = fn(value)

        return cast(T, x_mapped[key])

    record = cast(Any, value)

    return type(record)(
        **{
            field.name: _map_tensors(getattr(record, field.name), fn, x_mapped)
            for field in fields(record)
        }
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
    # Scanned along the last axis: CUDA's first-axis scan runs one serial thread per column
    return x_count.movedim(0, -1).cumsum(-1).movedim(-1, 0) - x_count


def get_unique_rep_row(x_inv: torch.Tensor, num_unique: int) -> torch.Tensor:
    """Find each unique value's first row: its representative.

    Parameters
    ----------
    x_inv : torch.Tensor
        (N,) long. Each row's unique value, as `torch.unique` returns it with `return_inverse`.
    num_unique : int
        The number of unique values.

    Returns
    -------
    torch.Tensor
        (num_unique,) long. Each unique value's representative row.
    """
    num_rows = x_inv.shape[0]
    x_row = torch.arange(num_rows, device=x_inv.device)

    return torch.full(
        (num_unique,), num_rows, dtype=torch.long, device=x_inv.device
    ).scatter_reduce_(0, x_inv, x_row, reduce="amin")


def get_plan_unique(
    np_ids: np.ndarray, num_plan_ids: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the plan IDs present in a forward, in plan order.

    The input is a 1-D int64 array of IDs in [0, num_plan_ids). Return the IDs present,
    their first positions in the input, and each input's index among the IDs present.
    """
    # A missing ID keeps the scatter's initial value, one past the last input position.
    x_ids = torch.from_numpy(np.ascontiguousarray(np_ids))
    np_first = get_unique_rep_row(x_ids, num_plan_ids).numpy()
    np_unique = np.flatnonzero(np_first < len(np_ids))
    np_rep = np_first[np_unique]

    # Reuse the table to look up the compact index of each ID present in this forward.
    np_first[np_unique] = np.arange(len(np_unique))

    return np_unique, np_rep, np_first[np_ids]


def dedup_rows(
    np_ids: np.ndarray, np_size: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Select one row per plan ID present in a forward and sum those IDs' sizes.

    Parameters
    ----------
    np_ids : np.ndarray
        (N,) int64. Each row's ID in the rollout plan.
    np_size : np.ndarray
        (U_plan, K) int64. Each plan ID's K sizes.
    device : torch.device
        The forward's device.

    Returns
    -------
    x_rep : torch.Tensor
        (U,) long. The first row for each ID present, in plan order.
    x_inv : torch.Tensor
        (N,) long. Each row's index among the U IDs present.
    np_size_sum : np.ndarray
        (K,) int64. The total sizes across the IDs present, counting each once.
    """
    np_unique, np_rep, np_inv = get_plan_unique(np_ids, len(np_size))
    num_unique = len(np_unique)

    # Both index arrays in one copy to the device, from pinned memory without blocking on CUDA
    x_buffer_cpu = torch.from_numpy(np.concatenate([np_rep, np_inv]))
    x_buffer = (x_buffer_cpu.pin_memory() if device.type == "cuda" else x_buffer_cpu).to(
        device, non_blocking=True
    )

    return x_buffer[:num_unique], x_buffer[num_unique:], np_size[np_unique].sum(0)


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
        x_collection_kind=torch.zeros(
            num_buffers, num_tokens, len(CollectionKind), dtype=torch.uint8
        ),
        x_entity_kind=torch.zeros(num_buffers, num_tokens, dtype=torch.int16),
        x_legal_selection=torch.zeros(num_buffers, num_tokens, dtype=torch.long),
        x_legal_target=torch.zeros(num_buffers, num_tokens, dtype=torch.long),
        x_needs_target=torch.zeros(num_buffers, num_tokens, dtype=torch.bool),
    )
    effect_pending = _allocate_sparse(num_buffers, buffer_size, ENCODING_DIM_SPARSE_EFFECT)
    x_total = torch.zeros(
        num_buffers, buffer_size, len(CollectionKind) + len(StateCollection), dtype=torch.int32
    )

    # Reserve memory for states
    return States(
        tokens=tokens,
        effect_pending=effect_pending,
        x_context=torch.zeros(
            num_buffers, buffer_size, ENCODING_DIM_CONTEXT, dtype=torch.float32
        ),
        x_map_feature=torch.zeros(
            num_buffers, buffer_size, MAP_HEIGHT, MAP_WIDTH, ENCODING_DIM_ROOM, dtype=torch.uint8
        ),
        x_map_topology=torch.zeros(
            num_buffers,
            buffer_size,
            MAP_HEIGHT,
            MAP_WIDTH,
            MAP_TOPOLOGY_CHANNELS,
            dtype=torch.uint8,
        ),
        x_legal_action_type=torch.zeros(
            num_buffers, buffer_size, NUM_ACTION_TYPES, dtype=torch.bool
        ),
        x_total=x_total,
        x_total_cpu=x_total,
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
    return slice_states(states_reserved, 0, len(states_reserved.x_context))


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
    
    return _cat_records(batch_states, {})


def _cat_records(values: list[Any], x_joined: dict[tuple[int, ...], torch.Tensor]) -> Any:
    first = values[0]
    if isinstance(first, torch.Tensor):
        key = tuple(id(value) for value in values)
        if key not in x_joined:
            x_joined[key] = torch.cat(values)

        return x_joined[key]
    
    return type(first)(
        **{
            field.name: _cat_records([getattr(v, field.name) for v in values], x_joined)
            for field in fields(first)
        }
    )


def to_device(states: States[Live], device: torch.device) -> States[Live]:
    """Make an independent snapshot, using one pinned transfer for CUDA.

    The device tensors are views of one copied byte buffer. CPU totals stay on the host, in
    their own small allocation for a device snapshot. On CPU they alias the snapshot's totals.
    No result aliases a worker's reusable storage.

    Parameters
    ----------
    states : States[Live]
        The source states to snapshot.
    device : torch.device
        The destination device.

    Returns
    -------
    States[Live]
        The states on the destination device, with their shapes and dtypes preserved, totals
        also available on the CPU, and storage independent of the source states.
    """
    # Get distinct payload tensors; the CPU totals travel separately when already mirrored
    x_payload = {id(x_tensor): x_tensor for x_tensor in iter_tensors(states)}
    if states.x_total_cpu is not states.x_total:
        del x_payload[id(states.x_total_cpu)]
    x_leaves = list(x_payload.values())

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
    x_buffer = x_buffer.to(device, non_blocking=True) if device.type != "cpu" else x_buffer

    # Get back each leaf Tensor from the buffer, and map it to `states`
    x_copied = {
        id(x_leaf): _bytes_as(x_buffer, x_leaf, idx_start)
        for x_leaf, idx_start in zip(x_leaves, idx_starts)
    }
    x_total_cpu = (
        x_copied[id(states.x_total)] if device.type == "cpu" else states.x_total_cpu.clone()
    )
    x_copied.setdefault(id(states.x_total_cpu), x_total_cpu)
    
    return replace(
        map_tensors(states, lambda x_tensor: x_copied[id(x_tensor)]), x_total_cpu=x_total_cpu
    )


def _bytes_as(x_buffer: torch.Tensor, x_like: torch.Tensor, idx_start: int) -> torch.Tensor:
    size = x_like.numel() * x_like.element_size()
    
    return x_buffer[idx_start : idx_start + size].view(x_like.dtype).view(x_like.shape)


def select_sparse[TStorage](
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


def select_owned_rows[TStorage](
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
        sparse=select_sparse(owned_rows.sparse, selector_rows, selector_entries),
        x_count=owned_rows.x_count[selector_tokens],
    )


def slice_states[TStorage](
    states: States[TStorage],
    idx_start: int,
    idx_stop: int,
) -> States[Live]:
    """View consecutive states and every row they own.

    Parameters
    ----------
    states : States[TStorage]
        The source states, live or one completed reserved buffer, with valid CPU totals.
    idx_start : int
        The first state, inclusive. Negative indexes count from the end of the batch.
    idx_stop : int
        The last state, exclusive. Bounds are clipped as in Python slicing.

    Returns
    -------
    States[Live]
        Views of the selected states and their descendants, sharing source storage.
    """
    idx_start, idx_stop, _ = slice(idx_start, idx_stop).indices(len(states.x_context))
    if idx_start > idx_stop:
        raise ValueError(
            f"`idx_start` ({idx_start}) must be smaller than or equal to `idx_stop` ({idx_stop})"
        )
    
    x_total_cpu = states.x_total_cpu
    selector_states = slice(idx_start, idx_stop)

    # Each collection's rows before the first state, and before the stop
    row_start = x_total_cpu[:selector_states.start].sum(0).tolist()
    row_stop = x_total_cpu[:selector_states.stop].sum(0).tolist()
    selectors = {
        collection: slice(row_start[collection], row_stop[collection])
        for collection in StateCollection
    }

    return _select_states(states, selector_states, selectors, x_total_cpu[selector_states])


def gather_states[TStorage](
    states: States[TStorage],
    indices: tuple[torch.Tensor, torch.Tensor],
) -> States[Live]:
    """Gather states and every row they own, preserving index order and duplicates.

    Parameters
    ----------
    states : States[TStorage]
        The source states, live or one completed reserved buffer, with valid CPU totals.
    indices : tuple[torch.Tensor, torch.Tensor]
        The same one-dimensional long indexes on the CPU and the source states' device, in
        that order. CPU states can use the same tensor twice. Indexes are used without
        transfers; negative indexes count from the end of the batch.

    Returns
    -------
    States[Live]
        Independent copies of the selected states and their descendants. The source is unchanged.

    Raises
    ------
    ValueError
        If the pair's first tensor is not on the CPU.
    IndexError
        If a gather index falls outside the source batch.
    """
    x_idx_state_cpu, x_idx_state = indices
    if x_idx_state_cpu.device.type != "cpu":
        raise ValueError("State indexes must be on the CPU")

    # The selected states' rows in every collection, totalled on the host
    x_total_cpu_selected = states.x_total_cpu[x_idx_state_cpu]
    num_rows = x_total_cpu_selected.sum(0).tolist()

    # Each state's rows in every collection, and where they start
    x_total = states.x_total.long()
    x_state_first_row = get_owner_first_row(x_total)

    # Each collection's source rows, in output order
    selectors = {
        collection: expand_ranges(
            x_state_first_row[x_idx_state, collection],
            x_total[x_idx_state, collection],
            num_rows[collection],
        )
        for collection in StateCollection
    }

    return _select_states(states, x_idx_state, selectors, x_total_cpu_selected)


def _select_states[TStorage](
    states: States[TStorage],
    selector_states: slice | torch.Tensor,
    selectors: Mapping[StateCollection, slice | torch.Tensor],
    x_total_cpu_selected: torch.Tensor,
) -> States[Live]:
    """Apply prepared state and collection selectors to every field.

    Parameters
    ----------
    states : States[TStorage]
        The source states, live or one completed reserved buffer.
    selector_states : slice | torch.Tensor
        The selected states, as a slice or indexes on the source device.
    selectors : Mapping[StateCollection, slice | torch.Tensor]
        Each collection's rows, in the selected state order.
    x_total_cpu_selected : torch.Tensor
        The selected states' totals on the CPU.

    Returns
    -------
    States[Live]
        The selected fields, retaining the shared CPU and device totals on CPU batches.
    """
    tokens = states.tokens
    selector_tokens = selectors[StateCollection.TOKENS]

    return States(
        tokens=Tokens(
            features=EncodedFeatures(
                sparse=select_sparse(
                    tokens.features.sparse, selector_tokens,
                    selectors[StateCollection.TOKEN_ENTRIES],
                ),
                effects=select_owned_rows(
                    tokens.features.effects, selector_tokens,
                    selectors[StateCollection.EFFECTS],
                    selectors[StateCollection.EFFECT_ENTRIES],
                ),
                modifiers=select_owned_rows(
                    tokens.features.modifiers, selector_tokens,
                    selectors[StateCollection.MODIFIERS],
                    selectors[StateCollection.MODIFIER_ENTRIES],
                ),
            ),
            x_count=tokens.x_count[selector_states],
            x_collection_kind=tokens.x_collection_kind[selector_tokens],
            x_entity_kind=tokens.x_entity_kind[selector_tokens],
            x_legal_selection=tokens.x_legal_selection[selector_tokens],
            x_legal_target=tokens.x_legal_target[selector_tokens],
            x_needs_target=tokens.x_needs_target[selector_tokens],
        ),
        effect_pending=select_sparse(
            states.effect_pending, selector_states, selectors[StateCollection.PENDING_ENTRIES]
        ),
        x_context=states.x_context[selector_states],
        x_map_feature=states.x_map_feature[selector_states],
        x_map_topology=states.x_map_topology[selector_states],
        x_legal_action_type=states.x_legal_action_type[selector_states],
        x_total=(
            x_total_cpu_selected
            if states.x_total.device.type == "cpu"
            else states.x_total[selector_states]
        ),
        x_total_cpu=x_total_cpu_selected,
    )


def expand_ranges(
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

    # Repeat each range's shift for each of its rows
    x_shift = x_shift.repeat_interleave(x_count, output_size=num_rows)

    # Add each output row's position to its shift to get its source row
    return x_shift + torch.arange(num_rows, device=x_start.device)
