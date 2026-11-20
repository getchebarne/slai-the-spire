from dataclasses import dataclass
from typing import final

import numpy as np

from src.rl.types import Reserved
from src.rl.types import Sparse


# Every row table's first row: the row with no entries
TABLE_ROW_EMPTY = 0


@final
@dataclass
class RowTable:
    """Sparse rows, each named by its index.

    A row's entries sit together in the entry arrays, in the form a `Sparse` stores. The arrays
    grow as rows come. An added row stays dense until the table is next read: the dense rows
    are then scanned together.
    """

    np_col: np.ndarray              # (entry capacity,) int16
    np_value: np.ndarray            # (entry capacity,) float32
    np_row_first_entry: np.ndarray  # (row capacity,) int64, each row's first entry
    np_count: np.ndarray            # (row capacity,) int64, each row's number of entries
    num_rows: int                   # The scanned rows
    num_entries: int                # The scanned rows' entries
    max_entries: int                # The most nonzero entries one row may hold
    rows_dense: list[np.ndarray]    # The rows added since the last scan, each (C,) float32


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
    _scan_rows_dense(row_table)

    np_count = row_table.np_count[np_table_row]
    num_rows = len(np_count)
    num_entries = int(np_count.sum())
    if num_rows > len(sparse_out.x_count):
        raise IndexError("Sparse row count exceeds the sparse_out's capacity")

    if num_entries > len(sparse_out.x_value):
        raise IndexError("Sparse entry count exceeds the sparse_out's capacity")

    # Each entry moves by its row's shift: row table entry = sparse_out entry + shift
    np_shift = row_table.np_row_first_entry[np_table_row] - (np.cumsum(np_count) - np_count)
    np_idx_entry = np.repeat(np_shift, np_count) + np.arange(num_entries)

    sparse_out.x_count[:num_rows].numpy()[:] = np_count
    sparse_out.x_col[:num_entries].numpy()[:] = row_table.np_col[np_idx_entry]
    sparse_out.x_value[:num_entries].numpy()[:] = row_table.np_value[np_idx_entry]


def make_row_table(max_entries: int) -> RowTable:
    """Make a row table that holds only the empty row.

    Parameters
    ----------
    max_entries : int
        The maximum number of nonzero feature entries allowed in one row.

    Returns
    -------
    RowTable
        The row table, its row TABLE_ROW_EMPTY the one with no entries.
    """
    return RowTable(
        np_col=np.zeros(0, dtype=np.int16),
        np_value=np.zeros(0, dtype=np.float32),
        np_row_first_entry=np.zeros(1, dtype=np.int64),
        np_count=np.zeros(1, dtype=np.int64),
        num_rows=1,
        num_entries=0,
        max_entries=max_entries,
        rows_dense=[],
    )


def drop_rows(row_table: RowTable) -> None:
    """Drop every row of a row table but the empty one.

    Whatever holds an index into the row table must be dropped first.

    Parameters
    ----------
    row_table : RowTable
        The row table.
    """
    row_table.num_rows = 1
    row_table.num_entries = 0
    row_table.rows_dense.clear()


def add_row(row_table: RowTable, np_row: np.ndarray) -> int:
    """Add a dense feature row to a row table.

    The row stays dense until the table is next read.

    Parameters
    ----------
    row_table : RowTable
        The row table.
    np_row : np.ndarray
        (C,) float32. The dense row, which the table keeps: the caller must not write it again.

    Returns
    -------
    int
        The row's index in the row table.
    """
    row_table.rows_dense.append(np_row)

    return row_table.num_rows + len(row_table.rows_dense) - 1


def _scan_rows_dense(row_table: RowTable) -> None:
    """Scan the rows added to a row table since its last scan into its entry arrays.

    Dense row r takes the index `num_rows + r`.

    Parameters
    ----------
    row_table : RowTable
        The row table.
    """
    if not row_table.rows_dense:
        return

    np_dense = np.array(row_table.rows_dense)  # (R, C)
    num_cols = np_dense.shape[-1]

    # Flat indexes of non-zero entries
    np_idx_nonzero = np.flatnonzero(np_dense != 0)

    # The rows and columns each non-zero entry flat-space corresponds to
    np_entry_row, np_entry_col = np.divmod(np_idx_nonzero, num_cols)

    # Number of non-zero entries each row has
    np_count = np.bincount(np_entry_row, minlength=len(np_dense))

    # Sanity checks
    max_nonzero_entries_of_row = int(np_count.max(initial=0))
    if max_nonzero_entries_of_row > row_table.max_entries:
        raise IndexError(
            f"Sparse row has {max_nonzero_entries_of_row} entries, exceeding its cap of "
            f"{row_table.max_entries}"
        )

    # Increase number of rows and entries counters
    stop_row = row_table.num_rows + len(np_count)
    stop_entry = row_table.num_entries + len(np_idx_nonzero)

    # Reserve space for the new rows
    row_table.np_row_first_entry = _reserve(row_table.np_row_first_entry, stop_row)
    row_table.np_count = _reserve(row_table.np_count, stop_row)
    row_table.np_col = _reserve(row_table.np_col, stop_entry)
    row_table.np_value = _reserve(row_table.np_value, stop_entry)

    # The entries go after the scanned ones, row after row
    row_table.np_col[row_table.num_entries : stop_entry] = np_entry_col
    np.take(
        np_dense.reshape(-1),
        np_idx_nonzero,
        out=row_table.np_value[row_table.num_entries : stop_entry],
    )
    np_row_first_entry = row_table.num_entries + np.cumsum(np_count) - np_count
    row_table.np_row_first_entry[row_table.num_rows : stop_row] = np_row_first_entry
    row_table.np_count[row_table.num_rows : stop_row] = np_count

    row_table.num_rows = stop_row
    row_table.num_entries = stop_entry
    row_table.rows_dense.clear()


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
