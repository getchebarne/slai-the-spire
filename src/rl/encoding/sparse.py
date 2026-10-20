import numpy as np
import torch

from src.rl.types import NUM_ACTION_TYPES
from src.rl.types import X_EffectPending
from src.rl.types import X_EntitySparse
from src.rl.types import X_OwnedSparse


def entity_sparse(
    np_rows: np.ndarray, np_slice_kind: np.ndarray, num_sparse: int, device: torch.device
) -> X_EntitySparse:
    """An entity block from its (B, W, D) rows, each with at most `num_sparse` nonzero
    columns, and their (B, W) slice kinds; its masks all False, for build_masks to fill"""
    np_column, np_value = _dense_to_sparse(np_rows, num_sparse)
    return X_EntitySparse(
        x_column=torch.from_numpy(np_column).to(device),
        x_value=torch.from_numpy(np_value).to(device),
        x_slice_kind=torch.from_numpy(np_slice_kind).to(device),
        x_legal=torch.zeros(
            *np_slice_kind.shape, NUM_ACTION_TYPES, dtype=torch.bool, device=device
        ),
        x_needs_target=torch.zeros(np_slice_kind.shape, dtype=torch.bool, device=device),
        batch_size=[np_rows.shape[0]],
    )


def owned_sparse(
    np_rows: np.ndarray,
    np_owner_slice_kind: np.ndarray,
    np_owner_rank: np.ndarray,
    num_sparse: int,
    device: torch.device,
) -> X_OwnedSparse:
    """A family of owned rows from its (B, R, D) rows, each with at most `num_sparse` nonzero
    columns, and their (B, R) owner keys"""
    np_column, np_value = _dense_to_sparse(np_rows, num_sparse)
    return X_OwnedSparse(
        x_column=torch.from_numpy(np_column).to(device),
        x_value=torch.from_numpy(np_value).to(device),
        x_owner_slice_kind=torch.from_numpy(np_owner_slice_kind).to(device),
        x_owner_rank=torch.from_numpy(np_owner_rank).to(device),
        batch_size=[np_rows.shape[0]],
    )


def effect_pending_sparse(
    np_rows: np.ndarray, num_sparse: int, device: torch.device
) -> X_EffectPending:
    """The pending effects from their (B, D) rows, each with at most `num_sparse` nonzero
    columns"""
    np_column, np_value = _dense_to_sparse(np_rows, num_sparse)
    return X_EffectPending(
        x_column=torch.from_numpy(np_column).to(device),
        x_value=torch.from_numpy(np_value).to(device),
        batch_size=[np_rows.shape[0]],
    )


def _dense_to_sparse(np_rows: np.ndarray, num_sparse: int) -> tuple[np.ndarray, np.ndarray]:
    """Each row of the (..., D) float32 `np_rows` as its nonzero (column, value) entries in
    column order, then (0, 0.0) entries: (..., K) int16 columns and float32 values, K the rows'
    most nonzero columns (at least one). A row with more than `num_sparse` raises an IndexError"""
    encoding_dim = np_rows.shape[-1]

    # Number of rows in `np_rows.reshape(-1, encoding_dim)`
    num_rows = np_rows.size // encoding_dim

    # Indices of non-zero elements in `np_rows.reshape(-1)`, in increasing order; F = number of
    # non-zero elements. Scanning a bool mask is far faster than scanning the floats themselves
    np_idx_flat_nz = np.flatnonzero(np_rows != 0)  # (F,)

    # `np_row`: row indices of non-zero elements in `np_rows.reshape(-1, encoding_dim)`
    # `np_col`: column indices of non-zero elements in `np_rows.reshape(-1, encoding_dim)`
    np_row, np_col = np.divmod(np_idx_flat_nz, encoding_dim)  # (F,), (F,)

    # Number of non-zero elements in each row
    np_nz_cnt = np.bincount(np_row, minlength=num_rows)  # (num_rows,)

    # Output width: the fullest row's non-zero count, at least one; `initial=0` covers a block
    # with zero rows. The `num_sparse` cap only makes an overflowing row raise below
    num_slots = max(min(int(np_nz_cnt.max(initial=0)), num_sparse), 1)

    # `np_idx_flat_nz` is sorted, so each row's non-zero elements form one run in it.
    # `np_end`: one past each run's last entry
    np_end = np.cumsum(np_nz_cnt)  # (num_rows,)

    # Subtracting each row's non-zero element count gives the index where its run starts
    np_start = np_end - np_nz_cnt  # (num_rows,)

    # Each non-zero element looks up its row's start
    np_nz_start = np_start[np_row]  # (F,)

    # `np.arange(F)` are the absolute positions of each non-zero element in `np_idx_flat_nz`.
    # Subtracting `np_nz_start` gives the slots they occupy in their corresponding row
    np_slot = np.arange(np_idx_flat_nz.shape[0]) - np_nz_start  # (F,)

    # Initialize arrays to store the compacted results; their zeros double as the (0, 0.0)
    # padding
    np_column = np.zeros((num_rows, num_slots), dtype=np.int16)  # (num_rows, num_slots)
    np_value = np.zeros((num_rows, num_slots), dtype=np.float32)  # (num_rows, num_slots)

    # Write each non-zero element's column and value to cell (its row, its slot)
    np_column[np_row, np_slot] = np_col
    np_value[np_row, np_slot] = np_rows.reshape(-1)[np_idx_flat_nz]  # Takes non-zero elements only

    # Reshape to (..., num_slots) and return
    shape = (*np_rows.shape[:-1], num_slots)
    return np_column.reshape(shape), np_value.reshape(shape)
