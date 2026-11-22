"""Construction of named feature channels and their column offsets."""

from collections.abc import Callable


def make_channel_allocator() -> tuple[list[tuple[str, int, int]], Callable[..., int]]:
    """Make a (layout, alloc) pair.

    `alloc(channel, width=1, nonzeros=1)` appends and returns the offset, so offsets, widths and
    the total all derive from one list.

    Returns
    -------
    layout : list[tuple[str, int, int]]
        Every allocated channel, its width and its most nonzero entries, in allocation order.
    alloc : Callable[..., int]
        The allocator over `layout`.
    """
    layout: list[tuple[str, int, int]] = []

    def alloc(channel: str, width: int = 1, nonzeros: int = 1) -> int:
        offset = sum(w for _, w, _ in layout)
        layout.append((channel, width, nonzeros))

        return offset

    return layout, alloc
