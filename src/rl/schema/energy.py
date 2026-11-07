"""Feature layout and static catalogs for energy rows."""

from src.rl.schema.layout import make_channel_allocator


ENERGY_MIN = 0
ENERGY_MAX = 15  # Ice Cream accumulates energy across turns


CHANNEL_LAYOUT_ENERGY, _alloc = make_channel_allocator()
OFF_ENERGY = _alloc("energy", ENERGY_MAX - ENERGY_MIN + 1)
OFF_ENERGY_SCALAR = _alloc("energy_scalar")

ENCODING_DIM_ENERGY = sum(w for _, w in CHANNEL_LAYOUT_ENERGY)
