import numpy as np
import torch

from src.rl.types import X_EffectPending
from src.rl.types import X_OwnedSparse
from src.rl.types import X_Tokens


def compile_sparse_tokens(
    np_rows: np.ndarray,
    np_token_kind: np.ndarray,
    np_legal: np.ndarray,
    np_needs_target: np.ndarray,
    np_node_idx: np.ndarray,
    np_count: np.ndarray,
    num_sparse: int,
    device: torch.device,
    x_effect_rows: X_OwnedSparse,
    x_modifier_rows: X_OwnedSparse,
) -> X_Tokens:
    """Compile the tokens, held sparse, and the rows they own.

    Parameters
    ----------
    np_rows : np.ndarray
        (N, D) float32. The tokens' rows, each with at most `num_sparse` nonzero columns.
    np_token_kind : np.ndarray
        (N,) int16. The tokens' token kinds.
    np_legal : np.ndarray
        (N, NUM_ACTION_TYPES) bool. The tokens' legal bits.
    np_needs_target : np.ndarray
        (N,) bool. The tokens' needs-target bits.
    np_node_idx : np.ndarray
        (N,) int16. The tokens' Map nodes; 0 on every token but a Room's.
    np_count : np.ndarray
        (B,) int64. Each state's count of tokens.
    num_sparse : int
        The most nonzero columns a row may hold.
    device : torch.device
        The device of the returned tensors.
    x_effect_rows : X_OwnedSparse
        The Effect rows the tokens own.
    x_modifier_rows : X_OwnedSparse
        The Modifier rows the tokens own.

    Returns
    -------
    X_Tokens
        The tokens, held sparse, and the rows they own.
    """
    np_column, np_value = _dense_to_sparse(np_rows, num_sparse)
    return X_Tokens(
        x_column=torch.from_numpy(np_column).to(device),
        x_value=torch.from_numpy(np_value).to(device),
        x_token_kind=torch.from_numpy(np_token_kind).to(device),
        x_legal=torch.from_numpy(np_legal).to(device),
        x_needs_target=torch.from_numpy(np_needs_target).to(device),
        x_node_idx=torch.from_numpy(np_node_idx).to(device),
        effect_rows=x_effect_rows,
        modifier_rows=x_modifier_rows,
        x_count=torch.from_numpy(np_count).to(device),
    )


def compile_sparse_owned(
    np_rows: np.ndarray, np_count: np.ndarray, num_sparse: int, device: torch.device
) -> X_OwnedSparse:
    """Compile an owned row kind's rows, held sparse.

    Parameters
    ----------
    np_rows : np.ndarray
        (M, D) float32. The rows, token after token, each with at most `num_sparse` nonzero
        columns.
    np_count : np.ndarray
        (N,) int16. Each token's count of them.
    num_sparse : int
        The most nonzero columns a row may hold.
    device : torch.device
        The device of the returned tensors.

    Returns
    -------
    X_OwnedSparse
        The rows, held sparse, and each token's count of them.
    """
    np_column, np_value = _dense_to_sparse(np_rows, num_sparse)
    return X_OwnedSparse(
        x_column=torch.from_numpy(np_column).to(device),
        x_value=torch.from_numpy(np_value).to(device),
        x_count=torch.from_numpy(np_count).to(device),
    )


def compile_sparse_effect_pending(
    np_rows: np.ndarray, np_count: np.ndarray, num_sparse: int, device: torch.device
) -> X_EffectPending:
    """Compile the pending Effects, held sparse.

    Parameters
    ----------
    np_rows : np.ndarray
        (P, D) float32. The pending Effects' rows, each with at most `num_sparse` nonzero
        columns.
    np_count : np.ndarray
        (B,) int64. Each state's count of them, 0 or 1.
    num_sparse : int
        The most nonzero columns a row may hold.
    device : torch.device
        The device of the returned tensors.

    Returns
    -------
    X_EffectPending
        The pending Effects, held sparse, and each state's count of them.
    """
    np_column, np_value = _dense_to_sparse(np_rows, num_sparse)
    return X_EffectPending(
        x_column=torch.from_numpy(np_column).to(device),
        x_value=torch.from_numpy(np_value).to(device),
        x_count=torch.from_numpy(np_count).to(device),
    )


def _dense_to_sparse(np_rows: np.ndarray, num_sparse: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert each row of `np_rows` to its nonzero (column, value) entries.

    Each row holds its entries in column order, then (0, 0.0) entries. K is the rows' most nonzero
    columns (at least one). A row with more than `num_sparse` raises an IndexError.

    Parameters
    ----------
    np_rows : np.ndarray
        (..., D) float32. The rows.
    num_sparse : int
        The most nonzero columns a row may hold.

    Returns
    -------
    np_column : np.ndarray
        (..., K) int16. Each row's entries' columns.
    np_value : np.ndarray
        (..., K) float32. Each row's entries' values.
    """
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
