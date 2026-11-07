"""Construction of named feature channels and their column offsets."""

from collections.abc import Callable


def make_channel_allocator() -> tuple[list[tuple[str, int]], Callable[..., int]]:
    """Make a (layout, alloc) pair.

    `alloc(channel, width=1)` appends and returns the offset, so offsets, widths and the total
    all derive from one list.

    Returns
    -------
    layout : list[tuple[str, int]]
        Every allocated channel and its width, in allocation order.
    alloc : Callable[..., int]
        The allocator over `layout`.
    """
    layout: list[tuple[str, int]] = []

    def alloc(channel: str, width: int = 1) -> int:
        offset = sum(w for _, w in layout)
        layout.append((channel, width))
        return offset

    return layout, alloc
