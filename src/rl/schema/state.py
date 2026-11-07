"""Feature layout and static catalogs for state rows."""

from src.rl.constants import MAX_MONSTERS
from src.rl.schema.card import ENCODING_DIM_CARD
from src.rl.schema.card import ENCODING_DIM_SPARSE_CARD
from src.rl.schema.character import ENCODING_DIM_CHARACTER
from src.rl.schema.character import ENCODING_DIM_SPARSE_CHARACTER
from src.rl.schema.health_block import ENCODING_DIM_HEALTH_BLOCK
from src.rl.schema.health_block import ENCODING_DIM_SPARSE_HEALTH_BLOCK
from src.rl.schema.monster import ENCODING_DIM_MONSTER
from src.rl.schema.monster import ENCODING_DIM_SPARSE_MONSTER
from src.rl.schema.potion import ENCODING_DIM_POTION
from src.rl.schema.potion import ENCODING_DIM_SPARSE_POTION
from src.rl.schema.relic import ENCODING_DIM_RELIC
from src.rl.schema.relic import ENCODING_DIM_SPARSE_RELIC
from src.rl.schema.shop import ENCODING_DIM_PRICE
from src.rl.schema.shop import ENCODING_DIM_SPARSE_PRICE
from src.rl.types import Tag


# The stasis slot columns: a one-hot over the Monsters' roster posts (Monster.slot)
ENCODING_DIM_STASIS_SLOT = MAX_MONSTERS
ENCODING_DIM_SPARSE_SLOT = 1

# The Tag columns: a Multi-Hot over the Tags
ENCODING_DIM_TAG = len(Tag)
ENCODING_DIM_SPARSE_TAG = ENCODING_DIM_TAG

# Each Entity kind's first column in the shared space; Event Options and Rooms have no columns
COL_START_CARD = 0
COL_START_RELIC = COL_START_CARD + ENCODING_DIM_CARD
COL_START_POTION = COL_START_RELIC + ENCODING_DIM_RELIC
COL_START_MONSTER = COL_START_POTION + ENCODING_DIM_POTION
COL_START_CHARACTER = COL_START_MONSTER + ENCODING_DIM_MONSTER

# The ranges shared across tokens of different Entity kinds
COL_START_HEALTH_BLOCK = COL_START_CHARACTER + ENCODING_DIM_CHARACTER
COL_START_PRICE = COL_START_HEALTH_BLOCK + ENCODING_DIM_HEALTH_BLOCK
COL_START_STASIS_SLOT = COL_START_PRICE + ENCODING_DIM_PRICE

# The range Cards, Relics and Potions share, outside their cached rows
COL_START_TAG = COL_START_STASIS_SLOT + ENCODING_DIM_STASIS_SLOT

# Total token encoding dimension
ENCODING_DIM_TOKEN = COL_START_TAG + ENCODING_DIM_TAG

# Most nonzero columns a token row holds
ENCODING_DIM_SPARSE_TOKEN = max(
    ENCODING_DIM_SPARSE_CARD
    + ENCODING_DIM_SPARSE_PRICE
    + ENCODING_DIM_SPARSE_SLOT
    + ENCODING_DIM_SPARSE_TAG,
    ENCODING_DIM_SPARSE_RELIC + ENCODING_DIM_SPARSE_PRICE + ENCODING_DIM_SPARSE_TAG,
    ENCODING_DIM_SPARSE_POTION + ENCODING_DIM_SPARSE_PRICE + ENCODING_DIM_SPARSE_TAG,
    ENCODING_DIM_SPARSE_MONSTER + ENCODING_DIM_SPARSE_HEALTH_BLOCK + ENCODING_DIM_SPARSE_SLOT,
    ENCODING_DIM_SPARSE_CHARACTER + ENCODING_DIM_SPARSE_HEALTH_BLOCK,
)
