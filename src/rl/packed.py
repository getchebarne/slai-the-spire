"""Selection, concatenation and device transfer of compact jagged game batches."""

from collections.abc import Callable
from collections.abc import Iterator
from dataclasses import fields, replace
from typing import Any

import torch
import torch.nn.functional as F

from src.rl.types import BatchLayout
from src.rl.types import Jagged
from src.rl.types import MinibatchPlan
from src.rl.types import Sparse
from src.rl.types import StateBatch
from src.rl.types import Tokens


def _offsets(x_lengths: torch.Tensor) -> torch.Tensor:
    return F.pad(x_lengths.cumsum(0, dtype=torch.long), (1, 0))


def build_batch_layout(batch: StateBatch, x_counts_cpu: torch.Tensor) -> BatchLayout:
    """Cache boundaries once for an immutable rollout, on the device and CPU as needed."""
    if x_counts_cpu.device.type != "cpu" or x_counts_cpu.shape != (
        len(batch.tokens.x_lengths),
        8,
    ):
        raise ValueError(
            "Batch counts must be a CPU tensor with one row and eight totals per state"
        )
    tokens = batch.tokens
    return BatchLayout(
        x_token_offsets=_offsets(tokens.x_lengths),
        x_effect_offsets=_offsets(tokens.rows.effects.x_lengths),
        x_modifier_offsets=_offsets(tokens.rows.modifiers.x_lengths),
        x_pending_offsets=_offsets(batch.pending.x_lengths),
        x_token_entry_offsets=_offsets(tokens.rows.features.x_lengths),
        x_effect_entry_offsets=_offsets(tokens.rows.effects.rows.x_lengths),
        x_modifier_entry_offsets=_offsets(tokens.rows.modifiers.rows.x_lengths),
        x_pending_entry_offsets=_offsets(batch.pending.rows.x_lengths),
        x_state_offsets_cpu=F.pad(x_counts_cpu.cumsum(0, dtype=torch.long), (0, 0, 1, 0)),
    )


def plan_minibatches(
    x_counts_cpu: torch.Tensor,
    minibatch_size: int,
    num_epochs: int,
    device: torch.device,
    generator: torch.Generator,
) -> list[list[MinibatchPlan]]:
    """Plan every epoch before training, transferring all shuffled state IDs together."""
    if x_counts_cpu.device.type != "cpu" or x_counts_cpu.ndim != 2 or x_counts_cpu.shape[1] != 8:
        raise ValueError("Minibatch counts must be a CPU tensor with eight totals per state")
    if minibatch_size <= 0 or num_epochs <= 0:
        raise ValueError("Minibatch size and epoch count must be positive")
    starts = range(0, len(x_counts_cpu), minibatch_size)
    x_order = torch.stack(
        [torch.randperm(len(x_counts_cpu), generator=generator) for _ in range(num_epochs)]
    )
    totals = [
        [
            x_counts_cpu[x_epoch[start : start + minibatch_size]].sum(0, dtype=torch.long).tolist()
            for start in starts
        ]
        for x_epoch in x_order
    ]
    x_order = (x_order.pin_memory() if device.type == "cuda" else x_order).to(
        device, non_blocking=True
    )
    return [
        [
            MinibatchPlan(x_epoch[start : start + minibatch_size], *total)
            for start, total in zip(starts, epoch_totals)
        ]
        for x_epoch, epoch_totals in zip(x_order, totals)
    ]


def select_segments(
    x_lengths: torch.Tensor, x_offsets: torch.Tensor, x_parents: torch.Tensor, total_rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return selected lengths and child indices, preserving parent order and duplicates.

    ``total_rows`` is planned on the CPU so the GPU gather needs no scalar readback.
    Lengths may be int16; index arithmetic and repeat counts use int64.
    """
    x_starts = x_offsets[x_parents]
    x_selected = x_lengths[x_parents]
    x_shifts = x_starts - (x_selected.cumsum(0, dtype=torch.long) - x_selected)
    x_rows = torch.arange(total_rows, device=x_lengths.device) + x_shifts.repeat_interleave(
        x_selected.long(), output_size=total_rows
    )
    return x_selected, x_rows


def take_states(batch: StateBatch, plan: MinibatchPlan, layout: BatchLayout) -> StateBatch:
    """Gather planned states and all their descendants using cached source boundaries."""
    x_state_ids = plan.x_state_ids
    x_lengths, x_token_rows = select_segments(
        batch.tokens.x_lengths, layout.x_token_offsets, x_state_ids, plan.tokens
    )
    return replace(
        _map_direct_tensors(batch, lambda x_tensor: x_tensor[x_state_ids]),
        tokens=Jagged(
            x_lengths=x_lengths,
            rows=take_tokens(batch.tokens.rows, x_token_rows, plan, layout),
        ),
        pending=_take_sparse_segments(
            batch.pending,
            layout.x_pending_offsets,
            x_state_ids,
            plan.pending,
            layout.x_pending_entry_offsets,
            plan.pending_entries,
        ),
    )


def take_tokens(
    tokens: Tokens, x_token_rows: torch.Tensor, plan: MinibatchPlan, layout: BatchLayout
) -> Tokens:
    """Gather token fields and the sparse rows owned by those tokens."""
    return replace(
        _map_direct_tensors(tokens, lambda x_tensor: x_tensor[x_token_rows]),
        features=take_sparse_rows(
            tokens.features, x_token_rows, layout.x_token_entry_offsets, plan.token_entries
        ),
        effects=_take_sparse_segments(
            tokens.effects,
            layout.x_effect_offsets,
            x_token_rows,
            plan.effects,
            layout.x_effect_entry_offsets,
            plan.effect_entries,
        ),
        modifiers=_take_sparse_segments(
            tokens.modifiers,
            layout.x_modifier_offsets,
            x_token_rows,
            plan.modifiers,
            layout.x_modifier_entry_offsets,
            plan.modifier_entries,
        ),
    )


def take_sparse_rows(
    rows: Sparse, x_indices: torch.Tensor, x_offsets: torch.Tensor, total_entries: int
) -> Sparse:
    x_lengths, x_entries = select_segments(rows.x_lengths, x_offsets, x_indices, total_entries)
    return Sparse(x_lengths, rows.x_columns[x_entries], rows.x_values[x_entries])


def _take_sparse_segments(
    jagged: Jagged[Sparse],
    x_offsets: torch.Tensor,
    x_parents: torch.Tensor,
    total_rows: int,
    x_entry_offsets: torch.Tensor,
    total_entries: int,
) -> Jagged[Sparse]:
    x_lengths, x_rows = select_segments(jagged.x_lengths, x_offsets, x_parents, total_rows)
    return Jagged(x_lengths, take_sparse_rows(jagged.rows, x_rows, x_entry_offsets, total_entries))


def slice_states(batch: StateBatch, start: int, stop: int, layout: BatchLayout) -> StateBatch:
    """View consecutive states and their descendants using CPU boundaries; bounds clip as slices."""
    start, stop, _ = slice(start, stop).indices(len(batch.tokens.x_lengths))
    state_rows = slice(start, stop)
    first, last = layout.x_state_offsets_cpu[[start, stop]].tolist()
    slices = [slice(left, right) for left, right in zip(first, last)]
    token_rows, effect_rows, modifier_rows, pending_rows = slices[:4]
    token_entries, effect_entries, modifier_entries, pending_entries = slices[4:]
    tokens = batch.tokens.rows

    def sparse_view(rows: Sparse, indices: slice, entries: slice) -> Sparse:
        return Sparse(rows.x_lengths[indices], rows.x_columns[entries], rows.x_values[entries])

    return replace(
        _map_direct_tensors(batch, lambda x_tensor: x_tensor[state_rows]),
        tokens=Jagged(
            batch.tokens.x_lengths[state_rows],
            replace(
                _map_direct_tensors(tokens, lambda x_tensor: x_tensor[token_rows]),
                features=sparse_view(tokens.features, token_rows, token_entries),
                effects=Jagged(
                    tokens.effects.x_lengths[token_rows],
                    sparse_view(tokens.effects.rows, effect_rows, effect_entries),
                ),
                modifiers=Jagged(
                    tokens.modifiers.x_lengths[token_rows],
                    sparse_view(tokens.modifiers.rows, modifier_rows, modifier_entries),
                ),
            ),
        ),
        pending=Jagged(
            batch.pending.x_lengths[state_rows],
            sparse_view(batch.pending.rows, pending_rows, pending_entries),
        ),
    )


def count_blocks(batch: StateBatch) -> torch.Tensor:
    """Per-state row totals, then entry totals: tokens, effects, modifiers, pending.

    The actor keeps these small counts on the CPU for minibatch size planning.
    """
    tokens = batch.tokens
    x_token_offsets = _offsets(tokens.x_lengths)

    # Compose state -> token -> owned-row boundaries, then each row's entry boundaries.
    effects = tokens.rows.effects
    modifiers = tokens.rows.modifiers
    pending = batch.pending
    x_effect_offsets = _offsets(effects.x_lengths)[x_token_offsets]
    x_modifier_offsets = _offsets(modifiers.x_lengths)[x_token_offsets]
    x_pending_offsets = _offsets(pending.x_lengths)

    return torch.stack(
        [
            tokens.x_lengths,
            x_effect_offsets.diff(),
            x_modifier_offsets.diff(),
            pending.x_lengths,
            _offsets(tokens.rows.features.x_lengths)[x_token_offsets].diff(),
            _offsets(effects.rows.x_lengths)[x_effect_offsets].diff(),
            _offsets(modifiers.rows.x_lengths)[x_modifier_offsets].diff(),
            _offsets(pending.rows.x_lengths)[x_pending_offsets].diff(),
        ],
        dim=1,
    )


def cat_states(batches: list[StateBatch]) -> StateBatch:
    """Concatenate compact worker batches or rollout steps field by field."""
    if not batches:
        raise ValueError("cat_states requires at least one batch")
    return _cat_records(batches)


def _cat_records(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(values)
    # Records, including Jagged, concatenate their fields independently.
    # Lengths need no rebasing, and each child table remains in its parents' order.
    return type(first)(
        **{
            field.name: _cat_records([getattr(v, field.name) for v in values])
            for field in fields(first)
        }
    )


def iter_tensors(value: Any) -> Iterator[torch.Tensor]:
    """Walk the tensor leaves of the data records in declaration order."""
    if isinstance(value, torch.Tensor):
        yield value
    else:
        for field in fields(value):
            yield from iter_tensors(getattr(value, field.name))


def map_tensors(value: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    """Rebuild data records with a function applied to each tensor leaf."""
    if isinstance(value, torch.Tensor):
        return fn(value)
    return type(value)(
        **{field.name: map_tensors(getattr(value, field.name), fn) for field in fields(value)}
    )


def _map_direct_tensors(value: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    """Transform a record's direct tensor fields, leaving its nested records to the caller."""
    return replace(
        value,
        **{
            field.name: fn(x_tensor)
            for field in fields(value)
            if isinstance(x_tensor := getattr(value, field.name), torch.Tensor)
        },
    )


def to_device(batch: StateBatch, device: torch.device) -> StateBatch:
    """Make an independent snapshot, using one pinned transfer for CUDA.

    Every result tensor is a view of the copied byte buffer. On CPU, the staging
    buffer itself owns the snapshot; no result aliases a worker's reusable storage.
    """
    tensors = list(iter_tensors(batch))
    starts, end = [], 0
    for x_tensor in tensors:
        starts.append(end)
        end += -(-x_tensor.numel() * x_tensor.element_size() // 8) * 8
    x_staging = torch.empty(end, dtype=torch.uint8, pin_memory=device.type == "cuda")
    for x_tensor, start in zip(tensors, starts):
        _bytes_as(x_staging, start, x_tensor).copy_(x_tensor)
    x_packed = x_staging.to(device, non_blocking=True) if device.type == "cuda" else x_staging
    pieces = iter(_bytes_as(x_packed, start, x_tensor) for x_tensor, start in zip(tensors, starts))
    return map_tensors(batch, lambda x_tensor: next(pieces))


def _bytes_as(x_buffer: torch.Tensor, start: int, x_like: torch.Tensor) -> torch.Tensor:
    size = x_like.numel() * x_like.element_size()
    return x_buffer[start : start + size].view(x_like.dtype).view(x_like.shape)
