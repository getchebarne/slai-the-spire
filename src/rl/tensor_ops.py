"""Operations on nested data records whose leaves are tensors."""

from collections.abc import Callable
from collections.abc import Iterator
from dataclasses import fields
from typing import Any

import torch


__all__ = [
    "iter_tensors",
    "map_tensors",
]


def iter_tensors(value: Any) -> Iterator[torch.Tensor]:
    """Walk the tensor leaves of the data records in declaration order.

    Parameters
    ----------
    value : Any
        A tensor or nested dataclass record whose leaves are tensors.

    Yields
    ------
    torch.Tensor
        Each tensor leaf, in field declaration order.
    """
    if isinstance(value, torch.Tensor):
        yield value
    else:
        for field in fields(value):
            yield from iter_tensors(getattr(value, field.name))


def map_tensors(value: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    """Rebuild data records with a function applied to each tensor leaf.

    Parameters
    ----------
    value : Any
        A tensor or nested dataclass record whose leaves are tensors.
    fn : Callable[[torch.Tensor], torch.Tensor]
        The function to apply to each tensor leaf.

    Returns
    -------
    Any
        The transformed tensor or rebuilt record, preserving record types and nesting.
    """
    if isinstance(value, torch.Tensor):
        return fn(value)
    return type(value)(
        **{field.name: map_tensors(getattr(value, field.name), fn) for field in fields(value)}
    )
