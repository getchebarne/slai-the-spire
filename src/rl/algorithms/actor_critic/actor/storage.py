"""The Actor's shared weight slots and publication signal."""

from collections.abc import Mapping
from dataclasses import dataclass
from multiprocessing.synchronize import Semaphore

import torch
import torch.multiprocessing as mp


@dataclass
class _WeightSlot:
    """One slot the Learner publishes weights into, with their version and curriculum position."""

    x_version: torch.Tensor  # () int64, in shared memory

    # In shared CPU memory: containers block sharing the GPU memory of the Learner's allocator
    weights: dict[str, torch.Tensor]
    x_ascension_current: torch.Tensor  # () float64, in shared memory


@dataclass
class _WeightBuffer:
    """The weights the Learner publishes and the Actor plays: two slots, a new-version signal."""

    # Two suffice: the Learner overwrites the older slot only after taking rollouts played on it
    weight_slots: list[_WeightSlot]  # Last two weights published by the Learner
    sph_new_version: Semaphore  # Released once per publish; the Actor acquires once per version


def allocate_buffer(weights: Mapping[str, torch.Tensor]) -> _WeightBuffer:
    """Return a weight buffer sized for `weights`, with both slots unpublished.

    Parameters
    ----------
    weights : Mapping[str, torch.Tensor]
        The Learner's state dictionary; its tensors supply the shapes and dtypes.

    Returns
    -------
    _WeightBuffer
        The buffer, in shared CPU memory.
    """
    return _WeightBuffer(
        weight_slots=[
            _WeightSlot(
                x_version=torch.tensor(-1).share_memory_(),
                weights={
                    name: torch.empty_like(x_weight, device="cpu").share_memory_()
                    for name, x_weight in weights.items()
                },
                x_ascension_current=torch.tensor(0.0, dtype=torch.float64).share_memory_(),
            )
            for _ in range(2)
        ],
        sph_new_version=mp.get_context("spawn").Semaphore(0),
    )


def write_weights(
    weight_buffer: _WeightBuffer,
    weights: Mapping[str, torch.Tensor],
    version: int,
    ascension_current: float,
) -> None:
    """Copy the weights into the older slot, then publish its version and curriculum position.

    Parameters
    ----------
    weight_buffer : _WeightBuffer
        The shared weight storage; its older slot is overwritten.
    weights : Mapping[str, torch.Tensor]
        The Learner's current state dictionary.
    version : int
        The number of updates these weights have had.
    ascension_current : float
        The curriculum position the Actor should play them at.
    """
    older = min(weight_buffer.weight_slots, key=lambda ws: int(ws.x_version))
    for name, x_weight in weights.items():
        # A copy to the CPU's pageable memory returns once done.
        older.weights[name].copy_(x_weight)
    older.x_ascension_current.fill_(ascension_current)
    older.x_version.fill_(version)
    weight_buffer.sph_new_version.release()
