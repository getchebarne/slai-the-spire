import numpy as np

from src.rl.encoding.buffers import SparseBuffer


def encode_sparse_into(
    destination: SparseBuffer, slot: int, np_dense: np.ndarray, max_entries: int
) -> None:
    """Write dense feature rows as flat nonzero entries, preserving empty rows.

    Columns stay in increasing order within each row. The caller's Jagged groups these rows;
    the sparse lengths count feature entries per row.

    Parameters
    ----------
    destination : SparseBuffer
        The reserved row and entry arrays. Only the selected slot is written.
    slot : int
        The destination's zero-based slot index.
    np_dense : np.ndarray
        (..., C). Dense features with C columns. Leading dimensions flatten into R rows.
    max_entries : int
        The maximum number of nonzero feature entries allowed in one row.
    """
    if not 0 <= slot < len(destination.x_sizes):
        raise IndexError("Slot index is outside the sparse buffer")

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
    if num_rows > destination.x_lengths.shape[1]:
        raise IndexError("Sparse row count exceeds the destination's capacity")
    if num_entries > destination.x_values.shape[1]:
        raise IndexError("Sparse entry count exceeds the destination's capacity")
    
    destination.x_lengths[slot, :num_rows].numpy()[:] = np_lens
    destination.x_columns[slot, :num_entries].numpy()[:] = np_cols
    np.take(np_dense.reshape(-1), np_pos, out=destination.x_values[slot, :num_entries].numpy())
    destination.x_sizes[slot, 0] = num_rows
    destination.x_sizes[slot, 1] = num_entries
