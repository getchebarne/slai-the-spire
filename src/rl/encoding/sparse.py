import numpy as np
import torch

from src.rl.types import Sparse


def encode_sparse(np_dense: np.ndarray, max_entries: int, device: torch.device) -> Sparse:
    """Encode dense feature rows as flat nonzero entries, preserving empty rows.

    Columns stay in increasing order within each row. The caller's Jagged groups these rows;
    the sparse lengths count feature entries per row.

    Parameters
    ----------
    np_dense : np.ndarray
        (..., C). Dense features with C columns. Leading dimensions flatten into R rows.
    max_entries : int
        The maximum number of nonzero feature entries allowed in one row.
    device : torch.device
        The device on which to place the output tensors.

    Returns
    -------
    Sparse
        `x_lengths`: (R,) int16, including zero for empty rows. `x_columns`: (F,) int16 and
        `x_values`: (F,), the F nonzero entries in row order. Values preserve the input dtype.
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
    
    return Sparse(
        x_lengths=torch.from_numpy(np_lens.astype(np.int16)).to(device),
        x_columns=torch.from_numpy(np_cols.astype(np.int16)).to(device),
        x_values=torch.from_numpy(np_dense.reshape(-1)[np_pos]).to(device),
    )
