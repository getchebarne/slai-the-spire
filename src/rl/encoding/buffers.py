"""Writable observation capacity and views of its completed slots."""

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
from src.rl.types import Jagged
from src.rl.types import Sparse
from src.rl.types import StateBatch
from src.rl.types import Tokens


__all__ = [
    "SparseBuffer",
    "StateBuffer",
    "allocate_buffer",
    "read_buffer",
]


@dataclass
class SparseBuffer:
    """Reserved row and feature-entry arrays, with independent live sizes for each slot.

    The arrays include unused capacity. Their populated prefixes become a Sparse record only
    when read from a completed observation slot.
    """

    x_lengths: torch.Tensor  # (slots, row capacity) int16
    x_columns: torch.Tensor  # (slots, entry capacity) int16
    x_values: torch.Tensor  # (slots, entry capacity) float32
    x_sizes: torch.Tensor  # (slots, 2) long, live (rows, entries)


@dataclass
class StateBuffer:
    """Reusable CPU destinations for encoding fixed batches of observations.

    Every tensor has a leading slot axis. Sparse tables own their live row and entry counts;
    the other token fields use token_features' live row count. A slot is readable only after
    its writer completes all fields and sets x_ready. Rewriting it invalidates earlier views.
    """

    x_token_lengths: torch.Tensor  # (slots, batch size) long
    token_features: SparseBuffer
    x_token_kind: torch.Tensor  # (slots, token capacity) int16
    x_legal_selection: torch.Tensor  # (slots, token capacity, NUM_ACTION_TYPES) bool
    x_legal_target: torch.Tensor  # (slots, token capacity, NUM_ACTION_TYPES) bool
    x_needs_target: torch.Tensor  # (slots, token capacity) bool
    x_node_idx: torch.Tensor  # (slots, token capacity) int16

    # Each token's owned rows, and each state's pending Effect
    x_effect_lengths: torch.Tensor  # (slots, token capacity) int16
    effects: SparseBuffer
    x_modifier_lengths: torch.Tensor  # (slots, token capacity) int16
    modifiers: SparseBuffer
    x_pending_lengths: torch.Tensor  # (slots, batch size) long
    pending: SparseBuffer

    x_context: torch.Tensor  # (slots, batch size, ENCODING_DIM_CONTEXT) float32
    x_map_grid: torch.Tensor  # (slots, batch size, MAP_HEIGHT, MAP_WIDTH, NUM_CHANNELS) float32
    x_legal_action_type: torch.Tensor  # (slots, batch size, NUM_ACTION_TYPES) bool
    x_ready: torch.Tensor  # (slots,) bool, set only after a complete write


def allocate_buffer(num_slots: int, batch_size: int) -> StateBuffer:
    """Allocate reusable CPU capacity with every slot initially unreadable.

    Parameters
    ----------
    num_slots : int
        The number of independent observation slots, at least one.
    batch_size : int
        The number of states written into each slot, including zero for an empty batch.

    Returns
    -------
    StateBuffer
        Ordinary CPU tensors sized for the maximum token, row and feature-entry counts.
        The caller chooses whether to place these tensors in shared memory.
    """
    if num_slots <= 0 or batch_size < 0:
        raise ValueError("`num_slots` must be positive and `batch_size` must be nonnegative")

    num_tokens = batch_size * CAP_TOKENS
    return StateBuffer(
        x_token_lengths=torch.zeros(num_slots, batch_size, dtype=torch.long),
        token_features=_allocate_sparse(num_slots, num_tokens, ENCODING_DIM_SPARSE_TOKEN),
        x_token_kind=torch.zeros(num_slots, num_tokens, dtype=torch.int16),
        x_legal_selection=torch.zeros(num_slots, num_tokens, NUM_ACTION_TYPES, dtype=torch.bool),
        x_legal_target=torch.zeros(num_slots, num_tokens, NUM_ACTION_TYPES, dtype=torch.bool),
        x_needs_target=torch.zeros(num_slots, num_tokens, dtype=torch.bool),
        x_node_idx=torch.zeros(num_slots, num_tokens, dtype=torch.int16),
        x_effect_lengths=torch.zeros(num_slots, num_tokens, dtype=torch.int16),
        effects=_allocate_sparse(num_slots, batch_size * MAX_EFFECTS, ENCODING_DIM_SPARSE_EFFECT),
        x_modifier_lengths=torch.zeros(num_slots, num_tokens, dtype=torch.int16),
        modifiers=_allocate_sparse(
            num_slots, batch_size * MAX_MODIFIERS, ENCODING_DIM_SPARSE_MODIFIER
        ),
        x_pending_lengths=torch.zeros(num_slots, batch_size, dtype=torch.long),
        pending=_allocate_sparse(num_slots, batch_size, ENCODING_DIM_SPARSE_EFFECT),
        x_context=torch.zeros(num_slots, batch_size, ENCODING_DIM_CONTEXT, dtype=torch.float32),
        x_map_grid=torch.zeros(
            num_slots, batch_size, MAP_HEIGHT, MAP_WIDTH, NUM_CHANNELS, dtype=torch.float32
        ),
        x_legal_action_type=torch.zeros(num_slots, batch_size, NUM_ACTION_TYPES, dtype=torch.bool),
        x_ready=torch.zeros(num_slots, dtype=torch.bool),
    )


def read_buffer(buffer: StateBuffer, slot: int) -> StateBatch:
    """View a completed slot's populated prefixes until that slot is written again.

    Parameters
    ----------
    buffer : StateBuffer
        Reusable CPU destinations whose slot has completed encoding.
    slot : int
        The zero-based index of the completed observation slot.

    Returns
    -------
    StateBatch
        Live record views sharing the slot's backing tensors. Retained observations must be
        copied before the slot is reused.

    Raises
    ------
    IndexError
        The slot index is outside the buffer.
    RuntimeError
        The slot has not completed encoding, or its last write failed.
    """
    if not 0 <= slot < len(buffer.x_ready):
        raise IndexError("Slot index is outside the observation buffer")
    if not bool(buffer.x_ready[slot]):
        raise RuntimeError("The observation slot has no completed encoding")

    features = _read_sparse(buffer.token_features, slot)
    num_tokens = len(features.x_lengths)
    return StateBatch(
        tokens=Jagged(
            x_lengths=buffer.x_token_lengths[slot],
            rows=Tokens(
                features=features,
                x_kind=buffer.x_token_kind[slot, :num_tokens],
                x_legal_selection=buffer.x_legal_selection[slot, :num_tokens],
                x_legal_target=buffer.x_legal_target[slot, :num_tokens],
                x_needs_target=buffer.x_needs_target[slot, :num_tokens],
                x_node_idx=buffer.x_node_idx[slot, :num_tokens],
                effects=Jagged(
                    x_lengths=buffer.x_effect_lengths[slot, :num_tokens],
                    rows=_read_sparse(buffer.effects, slot),
                ),
                modifiers=Jagged(
                    x_lengths=buffer.x_modifier_lengths[slot, :num_tokens],
                    rows=_read_sparse(buffer.modifiers, slot),
                ),
            ),
        ),
        pending=Jagged(
            x_lengths=buffer.x_pending_lengths[slot],
            rows=_read_sparse(buffer.pending, slot),
        ),
        x_context=buffer.x_context[slot],
        x_map_grid=buffer.x_map_grid[slot],
        x_legal_action_type=buffer.x_legal_action_type[slot],
    )


def _allocate_sparse(num_slots: int, rows: int, entries_per_row: int) -> SparseBuffer:
    """Allocate sparse capacity and its live sizes for every observation slot.

    Parameters
    ----------
    num_slots : int
        The number of slots sharing each buffer's leading axis.
    rows : int
        Each slot's maximum number of sparse rows.
    entries_per_row : int
        The maximum number of nonzero entries in a row.

    Returns
    -------
    SparseBuffer
        Ordinary CPU arrays with no live rows or entries yet.
    """
    entries = rows * entries_per_row
    return SparseBuffer(
        x_lengths=torch.zeros(num_slots, rows, dtype=torch.int16),
        x_columns=torch.zeros(num_slots, entries, dtype=torch.int16),
        x_values=torch.zeros(num_slots, entries, dtype=torch.float32),
        x_sizes=torch.zeros(num_slots, 2, dtype=torch.long),
    )


def _read_sparse(buffer: SparseBuffer, slot: int) -> Sparse:
    """View the populated rows and entries of a completed sparse table.

    Parameters
    ----------
    buffer : SparseBuffer
        Reserved sparse arrays with live sizes written by the encoder.
    slot : int
        The index of the completed observation slot.

    Returns
    -------
    Sparse
        Views of the populated row and entry prefixes.
    """
    rows, entries = buffer.x_sizes[slot].tolist()
    return Sparse(
        x_lengths=buffer.x_lengths[slot, :rows],
        x_columns=buffer.x_columns[slot, :entries],
        x_values=buffer.x_values[slot, :entries],
    )
