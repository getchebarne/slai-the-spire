import numpy as np
import torch

from src.rl.types import NUM_ACTION_TYPES
from src.rl.types import X_EffectPending
from src.rl.types import X_OwnedSparse
from src.rl.types import X_Tokens


def tokens_sparse(
    np_rows: np.ndarray,
    np_token_kind: np.ndarray,
    np_node_idx: np.ndarray,
    np_count: np.ndarray,
    num_sparse: int,
    device: torch.device,
    effects: X_OwnedSparse,
    modifiers: X_OwnedSparse,
) -> X_Tokens:
    """The tokens from their (N, D) rows, each with at most `num_sparse` nonzero columns, their
    (N,) token kinds and nodes, and each state's (B,) count of them; their masks all False, for
    build_masks to fill; and the rows they own"""
    np_column, np_value = _dense_to_sparse(np_rows, num_sparse)
    return X_Tokens(
        x_column=torch.from_numpy(np_column).to(device),
        x_value=torch.from_numpy(np_value).to(device),
        x_token_kind=torch.from_numpy(np_token_kind).to(device),
        x_legal=torch.zeros(len(np_rows), NUM_ACTION_TYPES, dtype=torch.bool, device=device),
        x_needs_target=torch.zeros(len(np_rows), dtype=torch.bool, device=device),
        x_node_idx=torch.from_numpy(np_node_idx).to(device),
        effects=effects,
        modifiers=modifiers,
        x_count=torch.from_numpy(np_count).to(device),
    )


def owned_sparse(
    np_rows: np.ndarray, np_count: np.ndarray, num_sparse: int, device: torch.device
) -> X_OwnedSparse:
    """A family of owned rows from its (M, D) rows, each with at most `num_sparse` nonzero
    columns, token after token, and each token's (N,) count of them"""
    np_column, np_value = _dense_to_sparse(np_rows, num_sparse)
    return X_OwnedSparse(
        x_column=torch.from_numpy(np_column).to(device),
        x_value=torch.from_numpy(np_value).to(device),
        x_count=torch.from_numpy(np_count).to(device),
    )


def effect_pending_sparse(
    np_rows: np.ndarray, np_count: np.ndarray, num_sparse: int, device: torch.device
) -> X_EffectPending:
    """The pending effects from their (P, D) rows, each with at most `num_sparse` nonzero
    columns, and each state's (B,) count of them, 0 or 1"""
    np_column, np_value = _dense_to_sparse(np_rows, num_sparse)
    return X_EffectPending(
        x_column=torch.from_numpy(np_column).to(device),
        x_value=torch.from_numpy(np_value).to(device),
        x_count=torch.from_numpy(np_count).to(device),
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
