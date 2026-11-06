"""Reusable worker storage, with published shapes for its live jagged batch."""

from dataclasses import dataclass, fields
from typing import Any

import torch

from src.rl.constants import MAP_HEIGHT
from src.rl.constants import MAP_WIDTH
from src.rl.constants import MAX_EFFECTS
from src.rl.constants import MAX_MODIFIERS
from src.rl.encoding.effect import ENCODING_DIM_SPARSE_EFFECT
from src.rl.encoding.energy import ENCODING_DIM_ENERGY
from src.rl.encoding.event import ENCODING_DIM_EVENT_META
from src.rl.encoding.focus import ENCODING_DIM_FOCUS
from src.rl.encoding.map_ import ENCODING_DIM_MAP_META
from src.rl.encoding.map_ import NUM_CHANNELS
from src.rl.encoding.modifier import ENCODING_DIM_SPARSE_MODIFIER
from src.rl.encoding.reward import ENCODING_DIM_REWARD_META
from src.rl.encoding.shop import ENCODING_DIM_SHOP_META
from src.rl.encoding.state import CAP_TOKENS
from src.rl.encoding.state import ENCODING_DIM_ASCENSION
from src.rl.encoding.state import ENCODING_DIM_SPARSE_TOKEN
from src.rl.types import NUM_ACTION_TYPES
from src.rl.types import Jagged
from src.rl.types import X_GameState
from src.rl.types import X_SparseEntries
from src.rl.types import X_SparseRows
from src.rl.types import X_Tokens


@dataclass
class SharedSparseRows:
    """Capacity for row lengths and their flat entries, not a populated Jagged."""

    lengths: torch.Tensor
    entries: X_SparseEntries
    max_entries_per_row: int


@dataclass
class SharedStateSlot:
    """One worker's capacity. Only `read_slot` exposes its populated rows as a batch."""

    token_lengths: torch.Tensor
    token_features: SharedSparseRows
    token_kind: torch.Tensor
    legal: torch.Tensor
    needs_target: torch.Tensor
    node_idx: torch.Tensor
    effect_lengths: torch.Tensor
    effects: SharedSparseRows
    modifier_lengths: torch.Tensor
    modifiers: SharedSparseRows
    pending_lengths: torch.Tensor
    pending: SharedSparseRows
    dense: dict[str, torch.Tensor]
    shapes: torch.Tensor  # (4, 2): live (rows, entries) of tokens, effects, modifiers, pending


def allocate_slot(batch_size: int) -> SharedStateSlot:
    """Allocate one worker's shared capacity."""
    return allocate_slots(1, batch_size)[0]


def allocate_slots(num_workers: int, batch_size: int) -> list[SharedStateSlot]:
    """Allocate each field once, with an explicit worker axis, and return disjoint slot views."""

    def shared(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros(num_workers, *shape, dtype=dtype).share_memory_()

    def sparse(rows: int, width: int) -> SharedSparseRows:
        return SharedSparseRows(
            shared(rows, dtype=torch.int16),
            X_SparseEntries(shared(rows * width, dtype=torch.int16), shared(rows * width)),
            width,
        )

    num_tokens = batch_size * CAP_TOKENS
    dense_shapes = {
        "x_energy": (ENCODING_DIM_ENERGY,),
        "x_map_grid": (MAP_HEIGHT, MAP_WIDTH, NUM_CHANNELS),
        "x_map_meta": (ENCODING_DIM_MAP_META,),
        "x_focus": (ENCODING_DIM_FOCUS,),
        "x_ascension": (ENCODING_DIM_ASCENSION,),
        "x_reward_meta": (ENCODING_DIM_REWARD_META,),
        "x_shop_meta": (ENCODING_DIM_SHOP_META,),
        "x_event_meta": (ENCODING_DIM_EVENT_META,),
    }
    storage = SharedStateSlot(
        token_lengths=shared(batch_size, dtype=torch.long),
        token_features=sparse(num_tokens, ENCODING_DIM_SPARSE_TOKEN),
        token_kind=shared(num_tokens, dtype=torch.int16),
        legal=shared(num_tokens, NUM_ACTION_TYPES, dtype=torch.bool),
        needs_target=shared(num_tokens, dtype=torch.bool),
        node_idx=shared(num_tokens, dtype=torch.int16),
        effect_lengths=shared(num_tokens, dtype=torch.int16),
        effects=sparse(batch_size * MAX_EFFECTS, ENCODING_DIM_SPARSE_EFFECT),
        modifier_lengths=shared(num_tokens, dtype=torch.int16),
        modifiers=sparse(batch_size * MAX_MODIFIERS, ENCODING_DIM_SPARSE_MODIFIER),
        pending_lengths=shared(batch_size, dtype=torch.long),
        pending=sparse(batch_size, ENCODING_DIM_SPARSE_EFFECT),
        dense={
            **{name: shared(batch_size, *shape) for name, shape in dense_shapes.items()},
            "x_legal_action_type": shared(batch_size, NUM_ACTION_TYPES, dtype=torch.bool),
        },
        shapes=shared(4, 2, dtype=torch.long),
    )
    return [_worker_view(storage, worker) for worker in range(num_workers)]


def _worker_view(storage: SharedStateSlot, worker: int) -> SharedStateSlot:
    def sliced(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value[worker]
        if isinstance(value, int):
            return value
        if isinstance(value, dict):
            return {name: tensor[worker] for name, tensor in value.items()}
        return type(value)(
            **{field.name: sliced(getattr(value, field.name)) for field in fields(value)}
        )

    return SharedStateSlot(
        **{field.name: sliced(getattr(storage, field.name)) for field in fields(storage)}
    )


def write_slot(slot: SharedStateSlot, batch: X_GameState) -> None:
    """Copy a batch's live prefixes, then publish their shapes before the worker replies.

    Every destination is checked before any write. Bytes outside the live shapes stay stale;
    readers never include them.
    """
    tokens = batch.x_tokens.rows
    sparse = [
        (slot.token_features, tokens.x_features),
        (slot.effects, tokens.x_effect_rows.rows),
        (slot.modifiers, tokens.x_modifier_rows.rows),
        (slot.pending, batch.x_effect_pending.rows),
    ]
    token_pairs = [
        (slot.token_kind, tokens.x_token_kind),
        (slot.legal, tokens.x_legal),
        (slot.needs_target, tokens.x_needs_target),
        (slot.node_idx, tokens.x_node_idx),
        (slot.effect_lengths, tokens.x_effect_rows.lengths),
        (slot.modifier_lengths, tokens.x_modifier_rows.lengths),
    ]
    pairs = [
        (slot.token_lengths, batch.x_tokens.lengths),
        *token_pairs,
        (slot.pending_lengths, batch.x_effect_pending.lengths),
        *((target, getattr(batch, name)) for name, target in slot.dense.items()),
    ]
    if (
        batch.x_tokens.lengths.shape != slot.token_lengths.shape
        or batch.x_effect_pending.lengths.shape != slot.pending_lengths.shape
        or any(getattr(batch, name).shape != target.shape for name, target in slot.dense.items())
    ):
        raise ValueError("A worker batch must contain exactly its slot's number of states")
    num_tokens = len(tokens.x_features.lengths)
    if any(len(source) != num_tokens for _, source in token_pairs):
        raise ValueError("Token fields and owned-row lengths must match the number of tokens")
    for target, source in pairs:
        if source.shape[1:] != target.shape[1:]:
            raise ValueError("A worker field's trailing shape does not match its storage")
    for target, source in sparse:
        entries = source.rows
        if entries.x_value.ndim != 1 or entries.x_column.shape != entries.x_value.shape:
            raise ValueError("Sparse columns and values must have the same one-dimensional shape")
        pairs.extend(
            [
                (target.lengths, source.lengths),
                (target.entries.x_column, entries.x_column),
                (target.entries.x_value, entries.x_value),
            ]
        )
    for target, source in pairs:
        if source.device.type != "cpu" or source.dtype != target.dtype:
            raise ValueError("A worker field must be on the CPU with its storage's dtype")
        if source.ndim != target.ndim or any(a > b for a, b in zip(source.shape, target.shape)):
            raise ValueError(
                f"Worker field shape {tuple(source.shape)} exceeds capacity {tuple(target.shape)}"
            )
    # Validate every relationship after dtype/rank checks and before touching shared storage.
    lengths_and_totals = [
        (batch.x_tokens.lengths, num_tokens),
        (tokens.x_effect_rows.lengths, len(tokens.x_effect_rows.rows.lengths)),
        (tokens.x_modifier_rows.lengths, len(tokens.x_modifier_rows.rows.lengths)),
        (batch.x_effect_pending.lengths, len(batch.x_effect_pending.rows.lengths)),
        *((source.lengths, len(source.rows.x_value)) for _, source in sparse),
    ]
    for lengths, total in lengths_and_totals:
        if bool((lengths < 0).any()) or int(lengths.sum(dtype=torch.long)) != total:
            raise ValueError("Jagged lengths must be nonnegative and sum to their live row count")
    for target, source in sparse:
        if bool((source.lengths > target.max_entries_per_row).any()):
            raise ValueError("A sparse row exceeds its entry capacity")
    shapes = torch.tensor(
        [[len(source.lengths), len(source.rows.x_value)] for _, source in sparse], dtype=torch.long
    )
    for target, source in pairs:
        target[tuple(slice(0, size) for size in source.shape)].copy_(source)
    slot.shapes.copy_(shapes)


def read_slot(slot: SharedStateSlot) -> X_GameState:
    """View the published batch after the worker replies, valid until its next write."""
    token_shape, effect_shape, modifier_shape, pending_shape = slot.shapes.tolist()
    num_tokens = token_shape[0]
    return X_GameState(
        x_tokens=Jagged(
            slot.token_lengths,
            X_Tokens(
                x_features=_live_rows(slot.token_features, token_shape),
                x_token_kind=slot.token_kind[:num_tokens],
                x_legal=slot.legal[:num_tokens],
                x_needs_target=slot.needs_target[:num_tokens],
                x_node_idx=slot.node_idx[:num_tokens],
                x_effect_rows=Jagged(
                    slot.effect_lengths[:num_tokens], _live_rows(slot.effects, effect_shape)
                ),
                x_modifier_rows=Jagged(
                    slot.modifier_lengths[:num_tokens], _live_rows(slot.modifiers, modifier_shape)
                ),
            ),
        ),
        x_effect_pending=Jagged(slot.pending_lengths, _live_rows(slot.pending, pending_shape)),
        **slot.dense,
    )


def _live_rows(storage: SharedSparseRows, shape: list[int]) -> X_SparseRows:
    rows, entries = shape
    return Jagged(
        storage.lengths[:rows],
        X_SparseEntries(storage.entries.x_column[:entries], storage.entries.x_value[:entries]),
    )
