"""Feature layout and static catalogs for shop rows."""

from src.rl.schema.layout import make_channel_allocator


CHANNEL_LAYOUT_PRICE, _alloc = make_channel_allocator()
OFF_SHOP_PRICE = _alloc("price")
OFF_SHOP_AFFORDABLE = _alloc("affordable")
OFF_SHOP_GOLD_GAP = _alloc("gold_gap")  # `abs(gold - price)`

ENCODING_DIM_PRICE = sum(w for _, w in CHANNEL_LAYOUT_PRICE)

# Most nonzero price columns a row holds: one per channel, each a scalar
ENCODING_DIM_SPARSE_PRICE = len(CHANNEL_LAYOUT_PRICE)

# Shop meta: the purge-service price triple + the purged flag
OFF_SHOP_PURGED = ENCODING_DIM_PRICE
ENCODING_DIM_SHOP_META = ENCODING_DIM_PRICE + 1
