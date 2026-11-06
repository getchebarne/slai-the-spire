import numpy as np
import torch

from src.rl.types import Jagged
from src.rl.types import X_SparseEntries
from src.rl.types import X_SparseRows
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
    x_effect_rows: Jagged[X_SparseRows],
    x_modifier_rows: Jagged[X_SparseRows],
) -> Jagged[X_Tokens]:
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
    x_effect_rows : Jagged[X_SparseRows]
        The Effect rows the tokens own.
    x_modifier_rows : Jagged[X_SparseRows]
        The Modifier rows the tokens own.

    Returns
    -------
    Jagged[X_Tokens]
        The tokens, held sparse, and the rows they own.
    """
    features = compile_sparse_rows(np_rows, np_count, num_sparse, device)
    return Jagged(
        lengths=features.lengths,
        rows=X_Tokens(
            x_features=features.rows,
            x_token_kind=torch.from_numpy(np_token_kind).to(device),
            x_legal=torch.from_numpy(np_legal).to(device),
            x_needs_target=torch.from_numpy(np_needs_target).to(device),
            x_node_idx=torch.from_numpy(np_node_idx).to(device),
            x_effect_rows=x_effect_rows,
            x_modifier_rows=x_modifier_rows,
        ),
    )


def compile_sparse_rows(
    np_rows: np.ndarray, np_count: np.ndarray, num_sparse: int, device: torch.device
) -> Jagged[X_SparseRows]:
    """Compile consecutive segments of sparse rows.

    Parameters
    ----------
    np_rows : np.ndarray
        (M, D) float32. The rows, segment after segment, each with at most `num_sparse`
        nonzero columns.
    np_count : np.ndarray
        (N,) integer. The number of rows in each segment.
    num_sparse : int
        The most nonzero columns a row may hold.
    device : torch.device
        The device of the returned tensors.

    Returns
    -------
    Jagged[X_SparseRows]
        The sparse rows and their segment lengths.
    """
    np_lengths, np_column, np_value = _dense_to_sparse(np_rows, num_sparse)
    return Jagged(
        lengths=torch.from_numpy(np_count).to(device),
        rows=Jagged(
            lengths=torch.from_numpy(np_lengths).to(device),
            rows=X_SparseEntries(
                x_column=torch.from_numpy(np_column).to(device),
                x_value=torch.from_numpy(np_value).to(device),
            ),
        ),
    )


def _dense_to_sparse(
    np_rows: np.ndarray, num_sparse: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert dense rows to lengths and consecutive nonzero (column, value) entries.

    Entries remain in row and column order. Empty rows have length zero and retain their row
    position. A row with more than `num_sparse` entries raises an IndexError.

    Parameters
    ----------
    np_rows : np.ndarray
        (..., D) float32. The rows.
    num_sparse : int
        The most nonzero columns a row may hold.

    Returns
    -------
    np_lengths : np.ndarray
        (N,) int16. Each flattened row's number of entries, including zero for empty rows.
    np_column : np.ndarray
        (F,) int16. All entries' columns, in row order.
    np_value : np.ndarray
        (F,) float32. All entries' values, in the same order.
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

    max_entries = int(np_nz_cnt.max(initial=0))
    if max_entries > num_sparse:
        raise IndexError(
            f"Sparse row has {max_entries} entries, exceeding its cap of {num_sparse}"
        )
    if max_entries > np.iinfo(np.int16).max:
        raise ValueError("Sparse row lengths exceed int16 capacity")
    return (
        np_nz_cnt.astype(np.int16),
        np_col.astype(np.int16),
        np_rows.reshape(-1)[np_idx_flat_nz],
    )
