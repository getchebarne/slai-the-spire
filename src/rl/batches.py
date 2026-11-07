"""Selection, concatenation and device transfer of compact jagged game batches."""

from collections.abc import Callable
from dataclasses import fields, replace
from typing import Any

import torch
import torch.nn.functional as F

from src.rl.tensor_ops import iter_tensors
from src.rl.tensor_ops import map_tensors
from src.rl.types import BatchLayout
from src.rl.types import Jagged
from src.rl.types import MinibatchPlan
from src.rl.types import Sparse
from src.rl.types import StateBatch
from src.rl.types import Tokens


__all__ = [
    "build_batch_layout",
    "cat_states",
    "count_blocks",
    "slice_states",
    "take_states",
    "to_device",
]


def _offsets(x_lengths: torch.Tensor) -> torch.Tensor:
    return F.pad(x_lengths.cumsum(0, dtype=torch.long), (1, 0))


def build_batch_layout(batch: StateBatch, x_counts_cpu: torch.Tensor) -> BatchLayout:
    """Cache boundaries once for an immutable batch, on the device and CPU as needed.

    Parameters
    ----------
    batch : StateBatch
        The source states whose rows and entries the boundaries describe.
    x_counts_cpu : torch.Tensor
        (B, 8) on the CPU. Each state's row and entry totals, in `count_blocks` order.

    Returns
    -------
    BatchLayout
        The source boundaries, valid while the batch's lengths remain unchanged.
    """
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


def _select_segments(
    x_lengths: torch.Tensor, x_offsets: torch.Tensor, x_parents: torch.Tensor, total_rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return selected lengths and child indices, preserving parent order and duplicates.

    ``total_rows`` is planned on the CPU so the GPU gather needs no scalar readback.
    Lengths may be int16; index arithmetic and repeat counts use int64.

    Parameters
    ----------
    x_lengths : torch.Tensor
        (P,) integer. The number of child rows in each source segment.
    x_offsets : torch.Tensor
        (P + 1,) long. The source segments' cumulative row boundaries.
    x_parents : torch.Tensor
        (S,) long. The segments to select, in output order.
    total_rows : int
        The sum of the selected segments' lengths, including repeated selections.

    Returns
    -------
    x_selected : torch.Tensor
        (S,). The selected segment lengths, preserving their dtype.
    x_rows : torch.Tensor
        (total_rows,) long. The source child rows, in selected segment order.
    """
    x_starts = x_offsets[x_parents]
    x_selected = x_lengths[x_parents]
    x_shifts = x_starts - (x_selected.cumsum(0, dtype=torch.long) - x_selected)
    x_rows = torch.arange(total_rows, device=x_lengths.device) + x_shifts.repeat_interleave(
        x_selected.long(), output_size=total_rows
    )
    return x_selected, x_rows


def take_states(batch: StateBatch, plan: MinibatchPlan, layout: BatchLayout) -> StateBatch:
    """Gather planned states and all their descendants using cached source boundaries.

    Parameters
    ----------
    batch : StateBatch
        The source states.
    plan : MinibatchPlan
        The state indices in output order and the selected row and entry totals.
    layout : BatchLayout
        The source batch's cached boundaries.

    Returns
    -------
    StateBatch
        The selected states and their descendants, preserving order and duplicate selections.
    """
    x_state_ids = plan.x_state_ids
    x_lengths, x_token_rows = _select_segments(
        batch.tokens.x_lengths, layout.x_token_offsets, x_state_ids, plan.tokens
    )
    return replace(
        _map_direct_tensors(batch, lambda x_tensor: x_tensor[x_state_ids]),
        tokens=Jagged(
            x_lengths=x_lengths,
            rows=_take_tokens(batch.tokens.rows, x_token_rows, plan, layout),
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


def _take_tokens(
    tokens: Tokens, x_token_rows: torch.Tensor, plan: MinibatchPlan, layout: BatchLayout
) -> Tokens:
    """Gather token fields and the sparse rows owned by those tokens.

    Parameters
    ----------
    tokens : Tokens
        The source token rows.
    x_token_rows : torch.Tensor
        (N,) long. The token rows to select, in output order.
    plan : MinibatchPlan
        The selected row and entry totals.
    layout : BatchLayout
        The source batch's cached boundaries.

    Returns
    -------
    Tokens
        The selected tokens, with their feature entries and owned Effect and Modifier rows.
    """
    return replace(
        _map_direct_tensors(tokens, lambda x_tensor: x_tensor[x_token_rows]),
        features=_take_sparse_rows(
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


def _take_sparse_rows(
    rows: Sparse, x_indices: torch.Tensor, x_offsets: torch.Tensor, total_entries: int
) -> Sparse:
    """Gather sparse rows and their entries in the selected row order.

    Parameters
    ----------
    rows : Sparse
        The source sparse rows.
    x_indices : torch.Tensor
        (R,) long. The source rows to select, in output order.
    x_offsets : torch.Tensor
        The source rows' cumulative entry boundaries.
    total_entries : int
        The number of entries in the selected rows, including repeated selections.

    Returns
    -------
    Sparse
        The selected rows, with their columns and values in row order.
    """
    x_lengths, x_entries = _select_segments(rows.x_lengths, x_offsets, x_indices, total_entries)
    return Sparse(x_lengths, rows.x_columns[x_entries], rows.x_values[x_entries])


def _take_sparse_segments(
    jagged: Jagged[Sparse],
    x_offsets: torch.Tensor,
    x_parents: torch.Tensor,
    total_rows: int,
    x_entry_offsets: torch.Tensor,
    total_entries: int,
) -> Jagged[Sparse]:
    x_lengths, x_rows = _select_segments(jagged.x_lengths, x_offsets, x_parents, total_rows)
    return Jagged(
        x_lengths, _take_sparse_rows(jagged.rows, x_rows, x_entry_offsets, total_entries)
    )


def slice_states(batch: StateBatch, start: int, stop: int, layout: BatchLayout) -> StateBatch:
    """View consecutive states and their descendants using CPU boundaries; bounds clip as slices.

    Parameters
    ----------
    batch : StateBatch
        The source states.
    start : int
        The first state's index, inclusive.
    stop : int
        The last boundary, exclusive.
    layout : BatchLayout
        The source batch's cached boundaries.

    Returns
    -------
    StateBatch
        Views of the selected states and descendants, sharing the source tensors' storage.
    """
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

    Parameters
    ----------
    batch : StateBatch
        The states whose row and entry totals to count.

    Returns
    -------
    torch.Tensor
        (B, 8) long on the batch's device. Tokens, Effects, Modifiers and pending rows, followed
        by the feature-entry totals of those same four blocks.
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
    """Concatenate compact worker batches or rollout steps field by field.

    Parameters
    ----------
    batches : list[StateBatch]
        At least one batch, in output order, with matching field dtypes and trailing shapes.

    Returns
    -------
    StateBatch
        The concatenated states and their descendants, preserving each batch's row order.
    """
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

    Parameters
    ----------
    batch : StateBatch
        The source states to snapshot.
    device : torch.device
        The destination device.

    Returns
    -------
    StateBatch
        The states on the destination device, with their shapes and dtypes preserved and
        storage independent of the source batch.
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
