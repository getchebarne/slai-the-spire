from dataclasses import dataclass

import torch

from src.rl.constants import CAP_TOKENS
from src.rl.constants import MAP_HEIGHT
from src.rl.constants import MAP_WIDTH
from src.rl.constants import MAX_EFFECTS
from src.rl.constants import MAX_MODIFIERS
from src.rl.constants import NUM_ACTION_TYPES
from src.rl.schema.context import ENCODING_DIM_CONTEXT
from src.rl.schema.effect import ENCODING_DIM_SPARSE_EFFECT
from src.rl.schema.map_ import NUM_CHANNELS
from src.rl.schema.modifier import ENCODING_DIM_SPARSE_MODIFIER
from src.rl.schema.state import ENCODING_DIM_SPARSE_TOKEN
from src.rl.tensor_ops import RecordLayout
from src.rl.tensor_ops import flatten_tensors
from src.rl.tensor_ops import unflatten_tensors
from src.rl.types import Jagged
from src.rl.types import Sparse
from src.rl.types import StateBatch
from src.rl.types import Tokens


@dataclass
class _StateSlot:
    """Raw shared capacity and the structure needed to view its populated prefixes.

    Buffers follow the record layout's leaf order. Only `read_slot` exposes a live batch,
    valid until the worker's next write; unused capacity never belongs to its records.
    """

    x_buffers: list[torch.Tensor]
    layout: RecordLayout
    x_sizes: torch.Tensor  # (number of leaves,) long, each buffer's live leading size
    batch_size: int


def allocate_slot(batch_size: int) -> _StateSlot:
    """Allocate one worker's shared capacity.

    Parameters
    ----------
    batch_size : int
        The number of games assigned to the worker.

    Returns
    -------
    _StateSlot
        The worker's reusable capacity, with every tensor in shared CPU memory.
    """
    return allocate_slots(1, batch_size)[0]


def allocate_slots(num_workers: int, batch_size: int) -> list[_StateSlot]:
    """Allocate each field once, with an explicit worker axis, and return disjoint slot views.

    Parameters
    ----------
    num_workers : int
        The number of workers that will share the allocation.
    batch_size : int
        The number of games assigned to each worker.

    Returns
    -------
    list[_StateSlot]
        One slot per worker, with disjoint views into shared CPU storage.
    """
    if num_workers <= 0 or batch_size <= 0:
        raise ValueError("`num_workers` and `batch_size` must be positive")

    x_buffers_by_leaf: dict[int, torch.Tensor] = {}

    def shared(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros(num_workers, *shape, dtype=dtype).share_memory_()

    def reserve(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        x_buffer = shared(*shape, dtype=dtype)
        x_empty = x_buffer[0, :0]
        x_buffers_by_leaf[id(x_empty)] = x_buffer
        return x_empty

    def sparse(rows: int, entries_per_row: int) -> Sparse:
        entries = rows * entries_per_row
        return Sparse(
            x_lengths=reserve(rows, dtype=torch.int16),
            x_columns=reserve(entries, dtype=torch.int16),
            x_values=reserve(entries),
        )

    # Reserve capacity through the live hierarchy; its empty views form a valid zero-state batch.
    num_tokens = batch_size * CAP_TOKENS
    prototype = StateBatch(
        tokens=Jagged(
            x_lengths=reserve(batch_size, dtype=torch.long),
            rows=Tokens(
                features=sparse(num_tokens, ENCODING_DIM_SPARSE_TOKEN),
                x_kind=reserve(num_tokens, dtype=torch.int16),
                x_legal_selection=reserve(num_tokens, NUM_ACTION_TYPES, dtype=torch.bool),
                x_legal_target=reserve(num_tokens, NUM_ACTION_TYPES, dtype=torch.bool),
                x_needs_target=reserve(num_tokens, dtype=torch.bool),
                x_node_idx=reserve(num_tokens, dtype=torch.int16),
                effects=Jagged(
                    x_lengths=reserve(num_tokens, dtype=torch.int16),
                    rows=sparse(batch_size * MAX_EFFECTS, ENCODING_DIM_SPARSE_EFFECT),
                ),
                modifiers=Jagged(
                    x_lengths=reserve(num_tokens, dtype=torch.int16),
                    rows=sparse(batch_size * MAX_MODIFIERS, ENCODING_DIM_SPARSE_MODIFIER),
                ),
            ),
        ),
        pending=Jagged(
            x_lengths=reserve(batch_size, dtype=torch.long),
            rows=sparse(batch_size, ENCODING_DIM_SPARSE_EFFECT),
        ),
        x_context=reserve(batch_size, ENCODING_DIM_CONTEXT),
        x_map_grid=reserve(batch_size, MAP_HEIGHT, MAP_WIDTH, NUM_CHANNELS),
        x_legal_action_type=reserve(batch_size, NUM_ACTION_TYPES, dtype=torch.bool),
    )

    # Match buffers by leaf identity, so constructor keyword order cannot change their order.
    x_leaves, layout = flatten_tensors(prototype)
    assert layout is not None
    x_buffers = [x_buffers_by_leaf[id(x_leaf)] for x_leaf in x_leaves]
    x_sizes = shared(len(x_buffers), dtype=torch.long)
    return [
        _StateSlot(
            x_buffers=[x_buffer[worker] for x_buffer in x_buffers],
            layout=layout,
            x_sizes=x_sizes[worker],
            batch_size=batch_size,
        )
        for worker in range(num_workers)
    ]


def write_slot(slot: _StateSlot, batch: StateBatch) -> None:
    """Copy live prefixes, then publish their shapes before the worker replies.

    Validate every destination before writing. Bytes outside the live shapes stay stale;
    readers never include them.

    Parameters
    ----------
    slot : _StateSlot
        The worker's reusable shared capacity.
    batch : StateBatch
        The worker's encoded states on the CPU, with the slot's batch size and field dtypes.
    """
    x_sources, layout = flatten_tensors(batch)
    if layout != slot.layout:
        raise ValueError("A worker batch's record structure does not match its storage")

    for x_target, x_source in zip(slot.x_buffers, x_sources):
        if x_source.device.type != "cpu" or x_source.dtype != x_target.dtype:
            raise ValueError("A worker field must be on the CPU with its storage's dtype")
        if x_source.ndim != x_target.ndim or x_source.shape[1:] != x_target.shape[1:]:
            raise ValueError("A worker field's rank or trailing shape does not match its storage")
        if len(x_source) > len(x_target):
            raise ValueError(
                f"Worker field shape {tuple(x_source.shape)} exceeds capacity {tuple(x_target.shape)}"
            )

    # Validate relationships after dtype/rank checks and before touching shared storage.
    tokens = batch.tokens.rows
    x_state_fields = [
        batch.tokens.x_lengths,
        batch.pending.x_lengths,
        batch.x_context,
        batch.x_map_grid,
        batch.x_legal_action_type,
    ]
    if any(len(x_field) != slot.batch_size for x_field in x_state_fields):
        raise ValueError("A worker batch must contain exactly its slot's number of states")
    num_tokens = len(tokens.features.x_lengths)
    x_token_fields = [
        tokens.x_kind,
        tokens.x_legal_selection,
        tokens.x_legal_target,
        tokens.x_needs_target,
        tokens.x_node_idx,
        tokens.effects.x_lengths,
        tokens.modifiers.x_lengths,
    ]
    if any(len(x_field) != num_tokens for x_field in x_token_fields):
        raise ValueError("Token fields and owned-row lengths must match the number of tokens")

    # Allocation and publication use the same per-row entry limits.
    sparse = [
        (tokens.features, ENCODING_DIM_SPARSE_TOKEN),
        (tokens.effects.rows, ENCODING_DIM_SPARSE_EFFECT),
        (tokens.modifiers.rows, ENCODING_DIM_SPARSE_MODIFIER),
        (batch.pending.rows, ENCODING_DIM_SPARSE_EFFECT),
    ]
    for source, _ in sparse:
        if source.x_columns.shape != source.x_values.shape:
            raise ValueError("Sparse columns and values must have the same one-dimensional shape")
    lengths_and_totals = [
        (batch.tokens.x_lengths, num_tokens),
        (tokens.effects.x_lengths, len(tokens.effects.rows.x_lengths)),
        (tokens.modifiers.x_lengths, len(tokens.modifiers.rows.x_lengths)),
        (batch.pending.x_lengths, len(batch.pending.rows.x_lengths)),
        *((source.x_lengths, len(source.x_values)) for source, _ in sparse),
    ]
    for x_lengths, total in lengths_and_totals:
        if bool((x_lengths < 0).any()) or int(x_lengths.sum(dtype=torch.long)) != total:
            raise ValueError("Jagged lengths must be nonnegative and sum to their live row count")
    for source, entry_cap in sparse:
        if bool((source.x_lengths > entry_cap).any()):
            raise ValueError("A sparse row exceeds its entry capacity")

    x_sizes = torch.tensor([len(x_source) for x_source in x_sources], dtype=torch.long)
    for x_target, x_source in zip(slot.x_buffers, x_sources):
        x_target[: len(x_source)].copy_(x_source)
    slot.x_sizes.copy_(x_sizes)


def read_slot(slot: _StateSlot) -> StateBatch:
    """View the published batch after the worker replies, valid until its next write.

    Parameters
    ----------
    slot : _StateSlot
        The worker's shared capacity, with its live shapes already published.

    Returns
    -------
    StateBatch
        Views of the populated prefixes, sharing the slot's CPU storage.
    """
    x_views = [x_buffer[:size] for x_buffer, size in zip(slot.x_buffers, slot.x_sizes.tolist())]
    return unflatten_tensors(x_views, slot.layout)
