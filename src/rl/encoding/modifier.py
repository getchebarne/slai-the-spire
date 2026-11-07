import math

import numpy as np
from slai import Modifier

from src.rl.encoding.scalars import get_sqrt_norm
from src.rl.schema.modifier import MODIFIER_KIND_TO_IDX
from src.rl.schema.modifier import OFF_MODIFIER_BUFF
from src.rl.schema.modifier import OFF_MODIFIER_KIND
from src.rl.schema.modifier import OFF_MODIFIER_OWNER
from src.rl.schema.modifier import OFF_MODIFIER_STACKS
from src.rl.schema.modifier import OFF_MODIFIER_STACKS_NEGATIVE
from src.rl.schema.modifier import OFF_MODIFIER_STACKS_OHE
from src.rl.schema.modifier import STACKS_MAX
from src.rl.schema.modifier import STACKS_MAX_OF_MODIFIER_KIND
from src.rl.types import ModifierOwner


def encode_modifier_row_into(modifier: Modifier, owner: ModifierOwner, out: np.ndarray) -> None:
    # Modifier kind
    out[OFF_MODIFIER_KIND + MODIFIER_KIND_TO_IDX[modifier.kind]] = 1.0

    # Buff or debuff
    out[OFF_MODIFIER_BUFF] = float(modifier.kind.is_buff)

    # Stacks: signed float + sign bit + magnitude-only sqrt buckets
    stacks_max = STACKS_MAX_OF_MODIFIER_KIND.get(modifier.kind, STACKS_MAX)
    out[OFF_MODIFIER_STACKS] = get_sqrt_norm(modifier.stacks, stacks_max)
    out[OFF_MODIFIER_STACKS_NEGATIVE] = float(modifier.stacks < 0)
    out[OFF_MODIFIER_STACKS_OHE + int(math.sqrt(min(abs(modifier.stacks), stacks_max)))] = 1.0

    # The combatant carrying this modifier
    out[OFF_MODIFIER_OWNER + owner] = 1.0
