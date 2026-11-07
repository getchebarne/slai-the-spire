import numpy as np
import torch

from src.rl.types import Sparse


def encode_sparse(dense: np.ndarray, max_entries: int, device: torch.device) -> Sparse:
    """Encode dense feature rows as flat nonzero entries, preserving empty rows.

    Columns stay in increasing order within each row. Ownership of these rows belongs to the
    caller's Jagged; lengths here count feature entries only. Exceeding the per-row capacity
    raises IndexError instead of silently truncating features.
    """
    num_columns = dense.shape[-1]
    num_rows = dense.size // num_columns
    positions = np.flatnonzero(dense != 0)
    rows, columns = np.divmod(positions, num_columns)
    lengths = np.bincount(rows, minlength=num_rows)
    largest = int(lengths.max(initial=0))
    if largest > max_entries:
        raise IndexError(f"Sparse row has {largest} entries, exceeding its cap of {max_entries}")
    if largest > np.iinfo(np.int16).max:
        raise ValueError("Sparse row lengths exceed int16 capacity")
    return Sparse(
        x_lengths=torch.from_numpy(lengths.astype(np.int16)).to(device),
        x_columns=torch.from_numpy(columns.astype(np.int16)).to(device),
        x_values=torch.from_numpy(dense.reshape(-1)[positions]).to(device),
    )
