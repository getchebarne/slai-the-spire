"""Feature layout and static catalogs for potion rows."""

from slai import PotionName
from slai import PotionRarity

from src.rl.schema.layout import make_channel_allocator
from src.rl.types import Tag


POTION_NAME_TO_IDX = {potion_name: int(potion_name) for potion_name in PotionName.members()}
POTION_RARITY_TO_IDX = {rarity: int(rarity) for rarity in PotionRarity.members()}


CHANNEL_LAYOUT_POTION, _alloc = make_channel_allocator()
OFF_POTION_NAME = _alloc("name", len(POTION_NAME_TO_IDX))
OFF_POTION_RARITY = _alloc("rarity", len(POTION_RARITY_TO_IDX))
OFF_POTION_REQUIRES_TARGET = _alloc("requires_target")
OFF_POTION_COMBAT_ONLY = _alloc("combat_only")
OFF_POTION_DOUBLED = _alloc(
    "doubled"
)  # Sacred Bark: its Effect rows already carry the doubled amounts

ENCODING_DIM_POTION = sum(w for _, w in CHANNEL_LAYOUT_POTION)

# Most nonzero columns a Potion row holds: one per channel, each a one-hot or a scalar
ENCODING_DIM_SPARSE_POTION = len(CHANNEL_LAYOUT_POTION)

# Each tagged Potion's Tags, by name: Sacred Bark's variant keeps them
TAGS_OF_POTION_NAME: dict[PotionName, tuple[Tag, ...]] = {
    PotionName.Cunning: (Tag.NINJA,),
    PotionName.Duplication: (Tag.SETUP,),
    PotionName.GamblersBrew: (Tag.DISCARD,),
    PotionName.LiquidMemories: (Tag.SETUP,),
    PotionName.Poison: (Tag.POISON,),
}
