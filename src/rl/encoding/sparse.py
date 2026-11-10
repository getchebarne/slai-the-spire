import numpy as np
import torch

from src.rl.types import SparseReserved


def encode_sparse_into(
    x_segment_lengths: torch.Tensor,
    destination: SparseReserved,
    np_segment_lengths: np.ndarray,
    np_dense: np.ndarray,
    max_entries: int,
) -> None:
    """Write each segment's row count and the dense feature rows as flat nonzero entries.

    The segment lengths count rows per segment; the sparse lengths count feature entries per row.

    Parameters
    ----------
    x_segment_lengths : torch.Tensor
        (segment capacity,) integer. Each segment's row count, written in place.
    destination : SparseReserved
        The reserved row and entry arrays of one table.
    np_segment_lengths : np.ndarray
        (S,). The number of rows in each of the S segments.
    np_dense : np.ndarray
        (..., C). Dense features with C columns. Leading dimensions flatten into R rows.
    max_entries : int
        The maximum number of nonzero feature entries allowed in one row.
    """
    encode_sparse_rows_into(destination, np_dense, max_entries)
    x_segment_lengths[: len(np_segment_lengths)].numpy()[:] = np_segment_lengths


def encode_sparse_rows_into(
    destination: SparseReserved, np_dense: np.ndarray, max_entries: int
) -> None:
    """Write dense feature rows as flat nonzero entries.

    Empty rows are preserved and columns stay in increasing order within each row.

    Parameters
    ----------
    destination : SparseReserved
        The reserved row and entry arrays of one table.
    np_dense : np.ndarray
        (..., C). Dense features with C columns. Leading dimensions flatten into R rows.
    max_entries : int
        The maximum number of nonzero feature entries allowed in one row.
    """
    num_cols = np_dense.shape[-1]
    num_rows = np_dense.size // num_cols

    # Flat indexes of non-zero entries
    np_pos = np.flatnonzero(np_dense != 0)

    # The rows and columns each non-zero entry flat-space corresponds to 
    np_rows, np_cols = np.divmod(np_pos, num_cols)

    # Number of non-zero entries each row has
    np_lens = np.bincount(np_rows, minlength=num_rows)

    # Sanity checks
    max_nonzero_entries_of_row = int(np_lens.max(initial=0))
    if max_nonzero_entries_of_row > max_entries:
        raise IndexError(f"Sparse row has {max_nonzero_entries_of_row} entries, exceeding its cap of {max_entries}")
    if max_nonzero_entries_of_row > np.iinfo(np.int16).max:
        raise ValueError("Sparse row np_lens exceed int16 capacity")
    num_entries = len(np_pos)
    if num_rows > len(destination.x_lengths):
        raise IndexError("Sparse row count exceeds the destination's capacity")
    if num_entries > len(destination.rows.x_values):
        raise IndexError("Sparse entry count exceeds the destination's capacity")
    
    destination.x_lengths[:num_rows].numpy()[:] = np_lens
    destination.rows.x_columns[:num_entries].numpy()[:] = np_cols
    np.take(np_dense.reshape(-1), np_pos, out=destination.rows.x_values[:num_entries].numpy())
