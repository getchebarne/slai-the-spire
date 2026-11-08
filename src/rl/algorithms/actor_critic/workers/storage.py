"""Shared observation storage and the worker publication boundary."""

from slai import Action
from slai import GameState

from src.rl.encoding import StateBuffer
from src.rl.encoding import allocate_buffer
from src.rl.encoding import encode_into
from src.rl.encoding import read_buffer
from src.rl.tensor_ops import iter_tensors
from src.rl.types import StateBatch


def allocate_store(num_workers: int, batch_size: int) -> StateBuffer:
    """Allocate the workers' observation capacity in shared CPU memory.

    Parameters
    ----------
    num_workers : int
        The number of workers that will share the allocation.
    batch_size : int
        The number of games assigned to each worker.

    Returns
    -------
    StateBuffer
        One shared store with a worker axis, initially unpublished.
    """
    if num_workers <= 0 or batch_size <= 0:
        raise ValueError("`num_workers` and `batch_size` must be positive")

    store = allocate_buffer(num_workers, batch_size)
    for x_tensor in iter_tensors(store):
        x_tensor.share_memory_()
    return store


def write_slot(
    store: StateBuffer,
    worker: int,
    batch_game_state: list[GameState],
    batch_legal_actions: list[list[Action]],
) -> None:
    """Encode observations directly into the worker's slot before it replies.

    The slot becomes unreadable when encoding begins and is published only after every field
    is complete. A failed encode leaves it unpublished; a later successful write can reuse it.

    Parameters
    ----------
    store : StateBuffer
        All workers' reusable shared capacity.
    worker : int
        The zero-based worker index whose slot receives the observations.
    batch_game_state : list[GameState]
        The worker's states, one for every game assigned to it.
    batch_legal_actions : list[list[Action]]
        Each state's legal actions, in the same order.
    """
    encode_into(store, worker, batch_game_state, batch_legal_actions)


def read_slot(store: StateBuffer, worker: int) -> StateBatch:
    """View the published batch after the worker replies, valid until its next write.

    Parameters
    ----------
    store : StateBuffer
        All workers' shared capacity, with this worker's observations already published.
    worker : int
        The zero-based worker index whose published slot is read.

    Returns
    -------
    StateBatch
        Views of the populated prefixes, sharing this worker's region of the store.
    """
    return read_buffer(store, worker)
