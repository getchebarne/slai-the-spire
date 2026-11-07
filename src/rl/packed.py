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


def _offsets(lengths: torch.Tensor) -> torch.Tensor:
    return F.pad(lengths.cumsum(0, dtype=torch.long), (1, 0))


def build_batch_layout(batch: StateBatch, counts_cpu: torch.Tensor) -> BatchLayout:
    """Cache boundaries once for an immutable rollout, on the device and CPU as needed."""
    if counts_cpu.device.type != "cpu" or counts_cpu.shape != (
        len(batch.tokens.lengths),
        8,
    ):
        raise ValueError(
            "Batch counts must be a CPU tensor with one row and eight totals per state"
        )
    tokens = batch.tokens
    return BatchLayout(
        token_offsets=_offsets(tokens.lengths),
        effect_offsets=_offsets(tokens.rows.effects.lengths),
        modifier_offsets=_offsets(tokens.rows.modifiers.lengths),
        pending_offsets=_offsets(batch.pending.lengths),
        token_entry_offsets=_offsets(tokens.rows.features.lengths),
        effect_entry_offsets=_offsets(tokens.rows.effects.rows.lengths),
        modifier_entry_offsets=_offsets(tokens.rows.modifiers.rows.lengths),
        pending_entry_offsets=_offsets(batch.pending.rows.lengths),
        state_offsets_cpu=F.pad(counts_cpu.cumsum(0, dtype=torch.long), (0, 0, 1, 0)),
    )


def plan_minibatches(
    counts_cpu: torch.Tensor,
    minibatch_size: int,
    num_epochs: int,
    device: torch.device,
    generator: torch.Generator,
) -> list[list[MinibatchPlan]]:
    """Plan every epoch before training, transferring all shuffled state IDs together."""
    if counts_cpu.device.type != "cpu" or counts_cpu.ndim != 2 or counts_cpu.shape[1] != 8:
        raise ValueError("Minibatch counts must be a CPU tensor with eight totals per state")
    if minibatch_size <= 0 or num_epochs <= 0:
        raise ValueError("Minibatch size and epoch count must be positive")
    starts = range(0, len(counts_cpu), minibatch_size)
    order = torch.stack(
        [torch.randperm(len(counts_cpu), generator=generator) for _ in range(num_epochs)]
    )
    totals = [
        [
            counts_cpu[epoch[start : start + minibatch_size]].sum(0, dtype=torch.long).tolist()
            for start in starts
        ]
        for epoch in order
    ]
    order = (order.pin_memory() if device.type == "cuda" else order).to(device, non_blocking=True)
    return [
        [
            MinibatchPlan(epoch[start : start + minibatch_size], *total)
            for start, total in zip(starts, epoch_totals)
        ]
        for epoch, epoch_totals in zip(order, totals)
    ]


def select_segments(
    lengths: torch.Tensor, offsets: torch.Tensor, parents: torch.Tensor, total_rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return selected lengths and child indices, preserving parent order and duplicates.

    ``total_rows`` is planned on the CPU so the GPU gather needs no scalar readback.
    Lengths may be int16; index arithmetic and repeat counts use int64.
    """
    starts = offsets[parents]
    selected = lengths[parents]
    shifts = starts - (selected.cumsum(0, dtype=torch.long) - selected)
    rows = torch.arange(total_rows, device=lengths.device) + shifts.repeat_interleave(
        selected.long(), output_size=total_rows
    )
    return selected, rows


def take_states(batch: StateBatch, plan: MinibatchPlan, layout: BatchLayout) -> StateBatch:
    """Gather planned states and all their descendants using cached source boundaries."""
    state_ids = plan.state_ids
    lengths, token_rows = select_segments(
        batch.tokens.lengths, layout.token_offsets, state_ids, plan.tokens
    )
    return replace(
        _map_direct_tensors(batch, lambda tensor: tensor[state_ids]),
        tokens=Jagged(
            lengths=lengths,
            rows=take_tokens(batch.tokens.rows, token_rows, plan, layout),
        ),
        pending=_take_sparse_segments(
            batch.pending,
            layout.pending_offsets,
            state_ids,
            plan.pending,
            layout.pending_entry_offsets,
            plan.pending_entries,
        ),
    )


def take_tokens(
    tokens: Tokens, token_rows: torch.Tensor, plan: MinibatchPlan, layout: BatchLayout
) -> Tokens:
    """Gather token fields and the sparse rows owned by those tokens."""
    return replace(
        _map_direct_tensors(tokens, lambda tensor: tensor[token_rows]),
        features=take_sparse_rows(
            tokens.features, token_rows, layout.token_entry_offsets, plan.token_entries
        ),
        effects=_take_sparse_segments(
            tokens.effects,
            layout.effect_offsets,
            token_rows,
            plan.effects,
            layout.effect_entry_offsets,
            plan.effect_entries,
        ),
        modifiers=_take_sparse_segments(
            tokens.modifiers,
            layout.modifier_offsets,
            token_rows,
            plan.modifiers,
            layout.modifier_entry_offsets,
            plan.modifier_entries,
        ),
    )


def take_sparse_rows(
    rows: Sparse, indices: torch.Tensor, offsets: torch.Tensor, total_entries: int
) -> Sparse:
    lengths, entries = select_segments(rows.lengths, offsets, indices, total_entries)
    return Sparse(lengths, rows.columns[entries], rows.values[entries])


def _take_sparse_segments(
    jagged: Jagged[Sparse],
    offsets: torch.Tensor,
    parents: torch.Tensor,
    total_rows: int,
    entry_offsets: torch.Tensor,
    total_entries: int,
) -> Jagged[Sparse]:
    lengths, rows = select_segments(jagged.lengths, offsets, parents, total_rows)
    return Jagged(lengths, take_sparse_rows(jagged.rows, rows, entry_offsets, total_entries))


def slice_states(batch: StateBatch, start: int, stop: int, layout: BatchLayout) -> StateBatch:
    """View consecutive states and their descendants using CPU boundaries; bounds clip as slices."""
    start, stop, _ = slice(start, stop).indices(len(batch.tokens.lengths))
    state_rows = slice(start, stop)
    first, last = layout.state_offsets_cpu[[start, stop]].tolist()
    slices = [slice(left, right) for left, right in zip(first, last)]
    token_rows, effect_rows, modifier_rows, pending_rows = slices[:4]
    token_entries, effect_entries, modifier_entries, pending_entries = slices[4:]
    tokens = batch.tokens.rows

    def sparse_view(rows: Sparse, indices: slice, entries: slice) -> Sparse:
        return Sparse(rows.lengths[indices], rows.columns[entries], rows.values[entries])

    return replace(
        _map_direct_tensors(batch, lambda tensor: tensor[state_rows]),
        tokens=Jagged(
            batch.tokens.lengths[state_rows],
            replace(
                _map_direct_tensors(tokens, lambda tensor: tensor[token_rows]),
                features=sparse_view(tokens.features, token_rows, token_entries),
                effects=Jagged(
                    tokens.effects.lengths[token_rows],
                    sparse_view(tokens.effects.rows, effect_rows, effect_entries),
                ),
                modifiers=Jagged(
                    tokens.modifiers.lengths[token_rows],
                    sparse_view(tokens.modifiers.rows, modifier_rows, modifier_entries),
                ),
            ),
        ),
        pending=Jagged(
            batch.pending.lengths[state_rows],
            sparse_view(batch.pending.rows, pending_rows, pending_entries),
        ),
    )


def count_blocks(batch: StateBatch) -> torch.Tensor:
    """Per-state row totals, then entry totals: tokens, effects, modifiers, pending.

    The actor keeps these small counts on the CPU for minibatch size planning.
    """
    tokens = batch.tokens
    token_offsets = _offsets(tokens.lengths)

    # Compose state -> token -> owned-row boundaries, then each row's entry boundaries.
    effects = tokens.rows.effects
    modifiers = tokens.rows.modifiers
    pending = batch.pending
    effect_offsets = _offsets(effects.lengths)[token_offsets]
    modifier_offsets = _offsets(modifiers.lengths)[token_offsets]
    pending_offsets = _offsets(pending.lengths)

    return torch.stack(
        [
            tokens.lengths,
            effect_offsets.diff(),
            modifier_offsets.diff(),
            pending.lengths,
            _offsets(tokens.rows.features.lengths)[token_offsets].diff(),
            _offsets(effects.rows.lengths)[effect_offsets].diff(),
            _offsets(modifiers.rows.lengths)[modifier_offsets].diff(),
            _offsets(pending.rows.lengths)[pending_offsets].diff(),
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
            field.name: fn(tensor)
            for field in fields(value)
            if isinstance(tensor := getattr(value, field.name), torch.Tensor)
        },
    )


def to_device(batch: StateBatch, device: torch.device) -> StateBatch:
    """Make an independent snapshot, using one pinned transfer for CUDA.

    Every result tensor is a view of the copied byte buffer. On CPU, the staging
    buffer itself owns the snapshot; no result aliases a worker's reusable storage.
    """
    tensors = list(iter_tensors(batch))
    starts, end = [], 0
    for tensor in tensors:
        starts.append(end)
        end += -(-tensor.numel() * tensor.element_size() // 8) * 8
    staging = torch.empty(end, dtype=torch.uint8, pin_memory=device.type == "cuda")
    for tensor, start in zip(tensors, starts):
        _bytes_as(staging, start, tensor).copy_(tensor)
    packed = staging.to(device, non_blocking=True) if device.type == "cuda" else staging
    pieces = iter(_bytes_as(packed, start, tensor) for tensor, start in zip(tensors, starts))
    return map_tensors(batch, lambda tensor: next(pieces))


def _bytes_as(buffer: torch.Tensor, start: int, like: torch.Tensor) -> torch.Tensor:
    size = like.numel() * like.element_size()
    return buffer[start : start + size].view(like.dtype).view(like.shape)
