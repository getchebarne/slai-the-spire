import numpy as np

from src.rl.types import OwnedRows
from src.rl.types import Reserved
from src.rl.types import Sparse


def encode_owned_rows_into(
    owned_rows: OwnedRows[Reserved], np_count: np.ndarray, np_dense: np.ndarray, max_entries: int
) -> None:
    """Write each token's number of owned rows and those rows as flat nonzero entries.

    `owned_rows.x_count` counts rows per token; `owned_rows.sparse.x_count` entries per row.

    Parameters
    ----------
    owned_rows : OwnedRows[Reserved]
        The reserved owner counts and sparse rows of one collection.
    np_count : np.ndarray
        (N,). Each token's number of owned rows.
    np_dense : np.ndarray
        (..., C). Dense features with C columns. Leading dimensions flatten into R rows.
    max_entries : int
        The maximum number of nonzero feature entries allowed in one row.
    """
    encode_sparse_rows_into(owned_rows.sparse, np_dense, max_entries)
    owned_rows.x_count[: len(np_count)].numpy()[:] = np_count


def encode_sparse_rows_into(
    sparse_out: Sparse[Reserved], np_dense: np.ndarray, max_entries: int
) -> None:
    """Write dense feature rows as flat nonzero entries.

    Empty rows are preserved and columns stay in increasing order within each row.

    Parameters
    ----------
    sparse_out : Sparse[Reserved]
        The reserved row and entry arrays of one table.
    np_dense : np.ndarray
        (..., C). Dense features with C columns. Leading dimensions flatten into R rows.
    max_entries : int
        The maximum number of nonzero feature entries allowed in one row.
    """
    num_cols = np_dense.shape[-1]
    num_rows = np_dense.size // num_cols

    # Flat indexes of non-zero entries
    np_idx_nonzero = np.flatnonzero(np_dense != 0)

    # The rows and columns each non-zero entry flat-space corresponds to 
    np_entry_row, np_entry_col = np.divmod(np_idx_nonzero, num_cols)

    # Number of non-zero entries each row has
    np_count = np.bincount(np_entry_row, minlength=num_rows)

    # Sanity checks
    max_nonzero_entries_of_row = int(np_count.max(initial=0))
    if max_nonzero_entries_of_row > max_entries:
        raise IndexError(f"Sparse row has {max_nonzero_entries_of_row} entries, exceeding its cap of {max_entries}")
    
    if max_nonzero_entries_of_row > np.iinfo(np.int16).max:
        raise ValueError("Sparse row entry counts exceed int16 capacity")
    
    num_entries = len(np_idx_nonzero)
    if num_rows > len(sparse_out.x_count):
        raise IndexError("Sparse row count exceeds the sparse_out's capacity")
    
    if num_entries > len(sparse_out.x_value):
        raise IndexError("Sparse entry count exceeds the sparse_out's capacity")
    
    sparse_out.x_count[:num_rows].numpy()[:] = np_count
    sparse_out.x_col[:num_entries].numpy()[:] = np_entry_col
    np.take(np_dense.reshape(-1), np_idx_nonzero, out=sparse_out.x_value[:num_entries].numpy())
