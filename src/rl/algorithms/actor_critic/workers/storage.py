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
from src.rl.tensor_ops import map_tensors
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

    x_token_lengths: torch.Tensor
    token_features: Sparse
    x_token_kind: torch.Tensor
    x_legal_selection: torch.Tensor
    x_legal_target: torch.Tensor
    x_needs_target: torch.Tensor
    x_node_idx: torch.Tensor
    x_effect_lengths: torch.Tensor
    effects: Sparse
    x_modifier_lengths: torch.Tensor
    modifiers: Sparse
    x_pending_lengths: torch.Tensor
    pending: Sparse
    x_context: torch.Tensor
    x_map_grid: torch.Tensor
    x_legal_action_type: torch.Tensor
    x_shapes: torch.Tensor  # (4, 2): live (rows, entries) of tokens, effects, modifiers, pending


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
        x_token_lengths=shared(batch_size, dtype=torch.long),
        token_features=sparse("token_features", num_tokens),
        x_token_kind=shared(num_tokens, dtype=torch.int16),
        x_legal_selection=shared(num_tokens, NUM_ACTION_TYPES, dtype=torch.bool),
        x_legal_target=shared(num_tokens, NUM_ACTION_TYPES, dtype=torch.bool),
        x_needs_target=shared(num_tokens, dtype=torch.bool),
        x_node_idx=shared(num_tokens, dtype=torch.int16),
        x_effect_lengths=shared(num_tokens, dtype=torch.int16),
        effects=sparse("effects", batch_size * MAX_EFFECTS),
        x_modifier_lengths=shared(num_tokens, dtype=torch.int16),
        modifiers=sparse("modifiers", batch_size * MAX_MODIFIERS),
        x_pending_lengths=shared(batch_size, dtype=torch.long),
        pending=sparse("pending", batch_size),
        x_context=shared(batch_size, ENCODING_DIM_CONTEXT),
        x_map_grid=shared(batch_size, MAP_HEIGHT, MAP_WIDTH, NUM_CHANNELS),
        x_legal_action_type=shared(batch_size, NUM_ACTION_TYPES, dtype=torch.bool),
        x_shapes=shared(4, 2, dtype=torch.long),
    )
    return [
        map_tensors(storage, lambda x_tensor: x_tensor[worker]) for worker in range(num_workers)
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
    tokens = batch.tokens.rows
    sparse = [
        ("token_features", tokens.features),
        ("effects", tokens.effects.rows),
        ("modifiers", tokens.modifiers.rows),
        ("pending", batch.pending.rows),
    ]
    token_pairs = [
        (slot.x_token_kind, tokens.x_kind),
        (slot.x_legal_selection, tokens.x_legal_selection),
        (slot.x_legal_target, tokens.x_legal_target),
        (slot.x_needs_target, tokens.x_needs_target),
        (slot.x_node_idx, tokens.x_node_idx),
        (slot.x_effect_lengths, tokens.effects.x_lengths),
        (slot.x_modifier_lengths, tokens.modifiers.x_lengths),
    ]
    state_pairs = [
        (slot.x_token_lengths, batch.tokens.x_lengths),
        (slot.x_pending_lengths, batch.pending.x_lengths),
        (slot.x_context, batch.x_context),
        (slot.x_map_grid, batch.x_map_grid),
        (slot.x_legal_action_type, batch.x_legal_action_type),
    ]
    if any(x_source.shape != x_target.shape for x_target, x_source in state_pairs):
        raise ValueError("A worker batch must contain exactly its slot's number of states")
    num_tokens = len(tokens.features.x_lengths)
    if any(len(x_source) != num_tokens for _, x_source in token_pairs):
        raise ValueError("Token fields and owned-row lengths must match the number of tokens")
    pairs = state_pairs + token_pairs
    for x_target, x_source in pairs:
        if x_source.shape[1:] != x_target.shape[1:]:
            raise ValueError("A worker field's trailing shape does not match its storage")
    for name, source in sparse:
        target = getattr(slot, name)
        if source.x_values.ndim != 1 or source.x_columns.shape != source.x_values.shape:
            raise ValueError("Sparse columns and values must have the same one-dimensional shape")
        pairs.extend(
            [
                (target.x_lengths, source.x_lengths),
                (target.x_columns, source.x_columns),
                (target.x_values, source.x_values),
            ]
        )
    for x_target, x_source in pairs:
        if x_source.device.type != "cpu" or x_source.dtype != x_target.dtype:
            raise ValueError("A worker field must be on the CPU with its storage's dtype")
        if x_source.ndim != x_target.ndim or any(
            a > b for a, b in zip(x_source.shape, x_target.shape)
        ):
            raise ValueError(
                f"Worker field shape {tuple(x_source.shape)} exceeds capacity {tuple(x_target.shape)}"
            )
    # Validate relationships after dtype/rank checks and before touching shared storage.
    lengths_and_totals = [
        (batch.tokens.x_lengths, num_tokens),
        (tokens.effects.x_lengths, len(tokens.effects.rows.x_lengths)),
        (tokens.modifiers.x_lengths, len(tokens.modifiers.rows.x_lengths)),
        (batch.pending.x_lengths, len(batch.pending.rows.x_lengths)),
        *((source.x_lengths, len(source.x_values)) for _, source in sparse),
    ]
    for x_lengths, total in lengths_and_totals:
        if bool((x_lengths < 0).any()) or int(x_lengths.sum(dtype=torch.long)) != total:
            raise ValueError("Jagged lengths must be nonnegative and sum to their live row count")
    for name, source in sparse:
        if bool((source.x_lengths > _ENTRY_CAPS[name]).any()):
            raise ValueError("A sparse row exceeds its entry capacity")
    x_shapes = torch.tensor(
        [[len(source.x_lengths), len(source.x_values)] for _, source in sparse], dtype=torch.long
    )
    for x_target, x_source in pairs:
        x_target[tuple(slice(0, size) for size in x_source.shape)].copy_(x_source)
    slot.x_shapes.copy_(x_shapes)


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
    token_shape, effect_shape, modifier_shape, pending_shape = slot.x_shapes.tolist()
    num_tokens = token_shape[0]
    return StateBatch(
        tokens=Jagged(
            slot.x_token_lengths,
            Tokens(
                features=_live_rows(slot.token_features, token_shape),
                x_kind=slot.x_token_kind[:num_tokens],
                x_legal_selection=slot.x_legal_selection[:num_tokens],
                x_legal_target=slot.x_legal_target[:num_tokens],
                x_needs_target=slot.x_needs_target[:num_tokens],
                x_node_idx=slot.x_node_idx[:num_tokens],
                effects=Jagged(
                    slot.x_effect_lengths[:num_tokens], _live_rows(slot.effects, effect_shape)
                ),
                modifiers=Jagged(
                    slot.x_modifier_lengths[:num_tokens],
                    _live_rows(slot.modifiers, modifier_shape),
                ),
            ),
        ),
        pending=Jagged(slot.x_pending_lengths, _live_rows(slot.pending, pending_shape)),
        x_context=slot.x_context,
        x_map_grid=slot.x_map_grid,
        x_legal_action_type=slot.x_legal_action_type,
    )


def _live_rows(storage: Sparse, shape: list[int]) -> Sparse:
    """View the populated rows and entries of sparse storage.

    Parameters
    ----------
    storage : Sparse
        The sparse arrays, including unused trailing capacity.
    shape : list[int]
        The published number of rows and number of entries, in that order.

    Returns
    -------
    Sparse
        Views of the populated prefixes, sharing the original storage.
    """
    rows, entries = shape
    return Sparse(
        storage.x_lengths[:rows], storage.x_columns[:entries], storage.x_values[:entries]
    )
