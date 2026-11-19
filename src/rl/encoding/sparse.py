from dataclasses import dataclass
from typing import final

import numpy as np

from src.rl.types import OwnedRows
from src.rl.types import Reserved
from src.rl.types import Sparse


# Every row table's first row: the row with no entries
ROW_EMPTY = 0


@final
@dataclass
class RowTable:
    """Distinct sparse rows, each kept once and named by its index.

    A row's entries sit together in the entry arrays, in the form a `Sparse` stores. The arrays
    grow as rows come. A write's own rows sit past the kept ones: they last until the next ones
    are set or a row is added.
    """

    np_col: np.ndarray        # (entry capacity,) int16
    np_value: np.ndarray      # (entry capacity,) float32
    np_row_start: np.ndarray  # (row capacity,) int64, each row's first entry
    np_count: np.ndarray      # (row capacity,) int64, each row's number of entries
    num_rows: int             # The kept rows
    num_entries: int          # The kept rows' entries

    # Each kept row, by its entries' bytes
    row_of_entries: dict[tuple[bytes, bytes], int]


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
    np_count, np_entry_col, np_idx_nonzero = _scan_rows(np_dense, max_entries)
    num_rows = len(np_count)
    num_entries = len(np_idx_nonzero)
    _check_capacity(sparse_out, num_rows, num_entries)

    sparse_out.x_count[:num_rows].numpy()[:] = np_count
    sparse_out.x_col[:num_entries].numpy()[:] = np_entry_col
    np.take(np_dense.reshape(-1), np_idx_nonzero, out=sparse_out.x_value[:num_entries].numpy())


def encode_table_rows_into(
    sparse_out: Sparse[Reserved], row_table: RowTable, np_table_row: np.ndarray
) -> None:
    """Write rows of a row table as flat nonzero entries.

    Parameters
    ----------
    sparse_out : Sparse[Reserved]
        The reserved row and entry arrays of one table.
    row_table : RowTable
        The row table that holds the rows.
    np_table_row : np.ndarray
        (R,) intp. The row table's row for each sparse_out row.
    """
    np_count = row_table.np_count[np_table_row]
    num_rows = len(np_count)
    num_entries = int(np_count.sum())
    _check_capacity(sparse_out, num_rows, num_entries)

    # Each entry's place in the row table: its row's shift from sparse_out, then its own place
    np_shift = row_table.np_row_start[np_table_row] - (np.cumsum(np_count) - np_count)
    np_idx_entry = np.repeat(np_shift, np_count) + np.arange(num_entries)

    sparse_out.x_count[:num_rows].numpy()[:] = np_count
    sparse_out.x_col[:num_entries].numpy()[:] = row_table.np_col[np_idx_entry]
    sparse_out.x_value[:num_entries].numpy()[:] = row_table.np_value[np_idx_entry]


def make_row_table() -> RowTable:
    """Make a row table that holds only the empty row.

    Returns
    -------
    RowTable
        The row table, its row ROW_EMPTY the one with no entries.
    """
    return RowTable(
        np_col=np.zeros(0, dtype=np.int16),
        np_value=np.zeros(0, dtype=np.float32),
        np_row_start=np.zeros(1, dtype=np.int64),
        np_count=np.zeros(1, dtype=np.int64),
        num_rows=1,
        num_entries=0,
        row_of_entries={},
    )


def clear_row_table(row_table: RowTable) -> None:
    """Drop every row of a row table but the empty one.

    Whatever holds an index into the row table must be dropped with it.

    Parameters
    ----------
    row_table : RowTable
        The row table.
    """
    row_table.num_rows = 1
    row_table.num_entries = 0
    row_table.row_of_entries.clear()


def add_row(row_table: RowTable, np_row: np.ndarray, max_entries: int) -> int:
    """Find a dense feature row's index in a row table, keeping the row the first time it comes.

    Parameters
    ----------
    row_table : RowTable
        The row table.
    np_row : np.ndarray
        (C,) float32. The dense row.
    max_entries : int
        The maximum number of nonzero feature entries allowed in the row.

    Returns
    -------
    int
        The row's index in the row table.
    """
    # The row sits past the kept ones until it proves new
    set_last_rows(row_table, np_row, max_entries)
    entry_stop = row_table.num_entries + int(row_table.np_count[row_table.num_rows])
    entries = (
        row_table.np_col[row_table.num_entries : entry_stop].tobytes(),
        row_table.np_value[row_table.num_entries : entry_stop].tobytes(),
    )
    row = row_table.row_of_entries.get(entries)
    if row is None:
        row = row_table.num_rows
        row_table.row_of_entries[entries] = row
        row_table.num_rows += 1
        row_table.num_entries = entry_stop

    return row


def set_last_rows(row_table: RowTable, np_dense: np.ndarray, max_entries: int) -> None:
    """Scan dense feature rows into a row table's last rows, past the kept ones.

    Row r takes the index `num_rows + r`.

    Parameters
    ----------
    row_table : RowTable
        The row table.
    np_dense : np.ndarray
        (..., C). Dense features with C columns. Leading dimensions flatten into R rows.
    max_entries : int
        The maximum number of nonzero feature entries allowed in one row.
    """
    np_count, np_entry_col, np_idx_nonzero = _scan_rows(np_dense, max_entries)
    row_stop = row_table.num_rows + len(np_count)
    entry_stop = row_table.num_entries + len(np_idx_nonzero)
    row_table.np_row_start = _reserve(row_table.np_row_start, row_stop)
    row_table.np_count = _reserve(row_table.np_count, row_stop)
    row_table.np_col = _reserve(row_table.np_col, entry_stop)
    row_table.np_value = _reserve(row_table.np_value, entry_stop)

    # The entries go after the kept ones, row after row
    np_row_start = row_table.num_entries + np.cumsum(np_count) - np_count
    row_table.np_row_start[row_table.num_rows : row_stop] = np_row_start
    row_table.np_count[row_table.num_rows : row_stop] = np_count
    row_table.np_col[row_table.num_entries : entry_stop] = np_entry_col
    np.take(
        np_dense.reshape(-1),
        np_idx_nonzero,
        out=row_table.np_value[row_table.num_entries : entry_stop],
    )


def _scan_rows(
    np_dense: np.ndarray, max_entries: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the nonzero entries of dense feature rows.

    Parameters
    ----------
    np_dense : np.ndarray
        (..., C). Dense features with C columns. Leading dimensions flatten into R rows.
    max_entries : int
        The maximum number of nonzero feature entries allowed in one row.

    Returns
    -------
    np_count : np.ndarray
        (R,) int64. Each row's number of entries.
    np_entry_col : np.ndarray
        (F,) int64. Each entry's column, the entries ordered by row then column.
    np_idx_nonzero : np.ndarray
        (F,) int64. Each entry's flat index in `np_dense`.
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

    return np_count, np_entry_col, np_idx_nonzero


def _check_capacity(sparse_out: Sparse[Reserved], num_rows: int, num_entries: int) -> None:
    """Check that `sparse_out` has room for the rows and entries to write.

    Parameters
    ----------
    sparse_out : Sparse[Reserved]
        The reserved row and entry arrays of one table.
    num_rows : int
        The number of rows to write.
    num_entries : int
        The number of entries to write.
    """
    if num_rows > len(sparse_out.x_count):
        raise IndexError("Sparse row count exceeds the sparse_out's capacity")
    
    if num_entries > len(sparse_out.x_value):
        raise IndexError("Sparse entry count exceeds the sparse_out's capacity")


def _reserve(np_array: np.ndarray, size: int) -> np.ndarray:
    """Make room for `size` elements.

    Parameters
    ----------
    np_array : np.ndarray
        (N,). The array.
    size : int
        The number of elements to hold.

    Returns
    -------
    np.ndarray
        (M,), M at least `size`. The array itself when it is long enough, else a copy at least
        twice as long, zeros past the array's own elements.
    """
    if size <= len(np_array):
        return np_array

    np_reserved = np.zeros(max(size, 2 * len(np_array)), dtype=np_array.dtype)
    np_reserved[: len(np_array)] = np_array
    return np_reserved
