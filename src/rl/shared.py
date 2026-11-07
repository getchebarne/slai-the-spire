"""Reusable worker storage, with published shapes for its live jagged batch."""

from dataclasses import dataclass

import torch

from src.rl.constants import MAP_HEIGHT
from src.rl.constants import MAP_WIDTH
from src.rl.constants import MAX_EFFECTS
from src.rl.constants import MAX_MODIFIERS
from src.rl.encoding.context import ENCODING_DIM_CONTEXT
from src.rl.encoding.effect import ENCODING_DIM_SPARSE_EFFECT
from src.rl.encoding.map_ import NUM_CHANNELS
from src.rl.encoding.modifier import ENCODING_DIM_SPARSE_MODIFIER
from src.rl.encoding.state import CAP_TOKENS
from src.rl.encoding.state import ENCODING_DIM_SPARSE_TOKEN
from src.rl.packed import map_tensors
from src.rl.types import NUM_ACTION_TYPES
from src.rl.types import Jagged
from src.rl.types import Sparse
from src.rl.types import StateBatch
from src.rl.types import Tokens


# Allocation and publication use the same per-row entry limits.
_ENTRY_CAPS = {
    "token_features": ENCODING_DIM_SPARSE_TOKEN,
    "effects": ENCODING_DIM_SPARSE_EFFECT,
    "modifiers": ENCODING_DIM_SPARSE_MODIFIER,
    "pending": ENCODING_DIM_SPARSE_EFFECT,
}


@dataclass
class _StateSlot:
    """Private capacity, not a live batch: Sparse arrays include unused trailing storage.

    Only `read_slot` exposes populated prefixes, valid until the worker's next write.
    """

    token_lengths: torch.Tensor
    token_features: Sparse
    token_kind: torch.Tensor
    legal: torch.Tensor
    needs_target: torch.Tensor
    node_idx: torch.Tensor
    effect_lengths: torch.Tensor
    effects: Sparse
    modifier_lengths: torch.Tensor
    modifiers: Sparse
    pending_lengths: torch.Tensor
    pending: Sparse
    context: torch.Tensor
    map_grid: torch.Tensor
    legal_action_type: torch.Tensor
    shapes: torch.Tensor  # (4, 2): live (rows, entries) of tokens, effects, modifiers, pending


def allocate_slot(batch_size: int) -> _StateSlot:
    """Allocate one worker's shared capacity."""
    return allocate_slots(1, batch_size)[0]


def allocate_slots(num_workers: int, batch_size: int) -> list[_StateSlot]:
    """Allocate each field once, with an explicit worker axis, and return disjoint slot views."""

    def shared(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros(num_workers, *shape, dtype=dtype).share_memory_()

    def sparse(name: str, rows: int) -> Sparse:
        entries = rows * _ENTRY_CAPS[name]
        return Sparse(
            shared(rows, dtype=torch.int16),
            shared(entries, dtype=torch.int16),
            shared(entries),
        )

    num_tokens = batch_size * CAP_TOKENS
    storage = _StateSlot(
        token_lengths=shared(batch_size, dtype=torch.long),
        token_features=sparse("token_features", num_tokens),
        token_kind=shared(num_tokens, dtype=torch.int16),
        legal=shared(num_tokens, NUM_ACTION_TYPES, dtype=torch.bool),
        needs_target=shared(num_tokens, dtype=torch.bool),
        node_idx=shared(num_tokens, dtype=torch.int16),
        effect_lengths=shared(num_tokens, dtype=torch.int16),
        effects=sparse("effects", batch_size * MAX_EFFECTS),
        modifier_lengths=shared(num_tokens, dtype=torch.int16),
        modifiers=sparse("modifiers", batch_size * MAX_MODIFIERS),
        pending_lengths=shared(batch_size, dtype=torch.long),
        pending=sparse("pending", batch_size),
        context=shared(batch_size, ENCODING_DIM_CONTEXT),
        map_grid=shared(batch_size, MAP_HEIGHT, MAP_WIDTH, NUM_CHANNELS),
        legal_action_type=shared(batch_size, NUM_ACTION_TYPES, dtype=torch.bool),
        shapes=shared(4, 2, dtype=torch.long),
    )
    return [map_tensors(storage, lambda tensor: tensor[worker]) for worker in range(num_workers)]


def write_slot(slot: _StateSlot, batch: StateBatch) -> None:
    """Copy live prefixes, then publish their shapes before the worker replies.

    Validate every destination before writing. Bytes outside the live shapes stay stale;
    readers never include them.
    """
    tokens = batch.tokens.rows
    sparse = [
        ("token_features", tokens.features),
        ("effects", tokens.effects.rows),
        ("modifiers", tokens.modifiers.rows),
        ("pending", batch.pending.rows),
    ]
    token_pairs = [
        (slot.token_kind, tokens.kind),
        (slot.legal, tokens.legal),
        (slot.needs_target, tokens.needs_target),
        (slot.node_idx, tokens.node_idx),
        (slot.effect_lengths, tokens.effects.lengths),
        (slot.modifier_lengths, tokens.modifiers.lengths),
    ]
    state_pairs = [
        (slot.token_lengths, batch.tokens.lengths),
        (slot.pending_lengths, batch.pending.lengths),
        (slot.context, batch.context),
        (slot.map_grid, batch.map_grid),
        (slot.legal_action_type, batch.legal_action_type),
    ]
    if any(source.shape != target.shape for target, source in state_pairs):
        raise ValueError("A worker batch must contain exactly its slot's number of states")
    num_tokens = len(tokens.features.lengths)
    if any(len(source) != num_tokens for _, source in token_pairs):
        raise ValueError("Token fields and owned-row lengths must match the number of tokens")
    pairs = state_pairs + token_pairs
    for target, source in pairs:
        if source.shape[1:] != target.shape[1:]:
            raise ValueError("A worker field's trailing shape does not match its storage")
    for name, source in sparse:
        target = getattr(slot, name)
        if source.values.ndim != 1 or source.columns.shape != source.values.shape:
            raise ValueError("Sparse columns and values must have the same one-dimensional shape")
        pairs.extend(
            [
                (target.lengths, source.lengths),
                (target.columns, source.columns),
                (target.values, source.values),
            ]
        )
    for target, source in pairs:
        if source.device.type != "cpu" or source.dtype != target.dtype:
            raise ValueError("A worker field must be on the CPU with its storage's dtype")
        if source.ndim != target.ndim or any(a > b for a, b in zip(source.shape, target.shape)):
            raise ValueError(
                f"Worker field shape {tuple(source.shape)} exceeds capacity {tuple(target.shape)}"
            )
    # Validate relationships after dtype/rank checks and before touching shared storage.
    lengths_and_totals = [
        (batch.tokens.lengths, num_tokens),
        (tokens.effects.lengths, len(tokens.effects.rows.lengths)),
        (tokens.modifiers.lengths, len(tokens.modifiers.rows.lengths)),
        (batch.pending.lengths, len(batch.pending.rows.lengths)),
        *((source.lengths, len(source.values)) for _, source in sparse),
    ]
    for lengths, total in lengths_and_totals:
        if bool((lengths < 0).any()) or int(lengths.sum(dtype=torch.long)) != total:
            raise ValueError("Jagged lengths must be nonnegative and sum to their live row count")
    for name, source in sparse:
        if bool((source.lengths > _ENTRY_CAPS[name]).any()):
            raise ValueError("A sparse row exceeds its entry capacity")
    shapes = torch.tensor(
        [[len(source.lengths), len(source.values)] for _, source in sparse], dtype=torch.long
    )
    for target, source in pairs:
        target[tuple(slice(0, size) for size in source.shape)].copy_(source)
    slot.shapes.copy_(shapes)


def read_slot(slot: _StateSlot) -> StateBatch:
    """View the published batch after the worker replies, valid until its next write."""
    token_shape, effect_shape, modifier_shape, pending_shape = slot.shapes.tolist()
    num_tokens = token_shape[0]
    return StateBatch(
        tokens=Jagged(
            slot.token_lengths,
            Tokens(
                features=_live_rows(slot.token_features, token_shape),
                kind=slot.token_kind[:num_tokens],
                legal=slot.legal[:num_tokens],
                needs_target=slot.needs_target[:num_tokens],
                node_idx=slot.node_idx[:num_tokens],
                effects=Jagged(
                    slot.effect_lengths[:num_tokens], _live_rows(slot.effects, effect_shape)
                ),
                modifiers=Jagged(
                    slot.modifier_lengths[:num_tokens], _live_rows(slot.modifiers, modifier_shape)
                ),
            ),
        ),
        pending=Jagged(slot.pending_lengths, _live_rows(slot.pending, pending_shape)),
        context=slot.context,
        map_grid=slot.map_grid,
        legal_action_type=slot.legal_action_type,
    )


def _live_rows(storage: Sparse, shape: list[int]) -> Sparse:
    rows, entries = shape
    return Sparse(storage.lengths[:rows], storage.columns[:entries], storage.values[:entries])
