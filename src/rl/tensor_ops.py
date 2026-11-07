"""Operations on nested data records whose leaves are tensors."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import is_dataclass
from typing import Any

import torch


__all__ = [
    "RecordLayout",
    "flatten_tensors",
    "iter_tensors",
    "map_tensors",
    "unflatten_tensors",
]


@dataclass(frozen=True)
class RecordLayout:
    """A record constructor and its fields in declaration order; None marks a tensor leaf."""

    record_type: type
    fields: tuple[tuple[str, RecordLayout | None], ...]


def flatten_tensors(value: Any) -> tuple[list[torch.Tensor], RecordLayout | None]:
    """Separate a nested tensor record into its leaves and derived structure.

    Parameters
    ----------
    value : Any
        A tensor or nested dataclass record whose leaves are tensors. Record fields must be
        accepted by their constructor as keyword arguments.

    Returns
    -------
    tuple[list[torch.Tensor], RecordLayout | None]
        The tensor references in field declaration order and their record structure. A tensor
        root has no record layout. Tensor data is not copied.

    Raises
    ------
    TypeError
        A value is neither a tensor nor a dataclass instance, or a field is excluded from its
        record's constructor.
    """
    x_leaves = []

    def visit(value: Any) -> RecordLayout | None:
        if isinstance(value, torch.Tensor):
            x_leaves.append(value)
            return None

        if not is_dataclass(value) or isinstance(value, type):
            raise TypeError("Tensor records must contain only dataclass instances and tensors")
        record_fields = fields(value)
        if any(not field.init for field in record_fields):
            raise TypeError("Tensor record fields must be accepted by their constructor")

        return RecordLayout(
            type(value),
            tuple((field.name, visit(getattr(value, field.name))) for field in record_fields),
        )

    layout = visit(value)
    return x_leaves, layout


def unflatten_tensors(x_leaves: Iterable[torch.Tensor], layout: RecordLayout | None) -> Any:
    """Reconstruct a tensor record from leaves and its derived structure.

    Parameters
    ----------
    x_leaves : Iterable[torch.Tensor]
        The tensor references in the layout's field declaration order.
    layout : RecordLayout | None
        The structure returned by flatten_tensors, or None for a tensor root.

    Returns
    -------
    Any
        The rebuilt record or tensor, sharing the supplied tensor objects.

    Raises
    ------
    TypeError
        A leaf is not a tensor.
    ValueError
        The number of leaves does not match the record structure.
    """
    x_leaves = list(x_leaves)
    if any(not isinstance(x_leaf, torch.Tensor) for x_leaf in x_leaves):
        raise TypeError("Tensor record leaves must be tensors")
    leaves = iter(x_leaves)

    def rebuild(layout: RecordLayout | None) -> Any:
        if layout is None:
            try:
                return next(leaves)
            except StopIteration:
                raise ValueError("Too few tensor leaves for the record layout") from None

        return layout.record_type(
            **{name: rebuild(child_layout) for name, child_layout in layout.fields}
        )

    value = rebuild(layout)
    if next(leaves, None) is not None:
        raise ValueError("Too many tensor leaves for the record layout")
    return value


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
