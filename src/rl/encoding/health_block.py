import numpy as np

from src.rl.constants import BLOCK_MAX_ENTITY
from src.rl.constants import HEALTH_MAX_ENTITY
from src.rl.schema.health_block import BLOCK_MIN
from src.rl.schema.health_block import HEALTH_MIN
from src.rl.schema.health_block import LINEAR_SQRT_THRESHOLD
from src.rl.schema.health_block import OFF_HEALTH_BLOCK_BLOCK
from src.rl.schema.health_block import OFF_HEALTH_BLOCK_BLOCK_SCALAR
from src.rl.schema.health_block import OFF_HEALTH_BLOCK_HEALTH
from src.rl.schema.health_block import OFF_HEALTH_BLOCK_HEALTH_SCALAR
from src.rl.schema.health_block import OFF_HEALTH_BLOCK_TOTAL
from src.rl.schema.health_block import OFF_HEALTH_BLOCK_TOTAL_SCALAR
from src.rl.schema.health_block import TOTAL_MAX
from src.rl.schema.health_block import TOTAL_MIN
from src.rl.schema.scalars import get_piecewise_bucket


def encode_health_block_into(health: int, block: int, out: np.ndarray) -> None:
    total = health + block

    health_bucket = get_piecewise_bucket(
        health, HEALTH_MIN, HEALTH_MAX_ENTITY, LINEAR_SQRT_THRESHOLD
    )
    out[OFF_HEALTH_BLOCK_HEALTH + health_bucket] = 1.0
    block_bucket = get_piecewise_bucket(
        block, BLOCK_MIN, BLOCK_MAX_ENTITY, LINEAR_SQRT_THRESHOLD
    )
    out[OFF_HEALTH_BLOCK_BLOCK + block_bucket] = 1.0
    total_bucket = get_piecewise_bucket(total, TOTAL_MIN, TOTAL_MAX, LINEAR_SQRT_THRESHOLD)
    out[OFF_HEALTH_BLOCK_TOTAL + total_bucket] = 1.0

    # Scalars
    out[OFF_HEALTH_BLOCK_HEALTH_SCALAR] = min(health / HEALTH_MAX_ENTITY, 1.0)
    out[OFF_HEALTH_BLOCK_BLOCK_SCALAR] = min(block / BLOCK_MAX_ENTITY, 1.0)
    out[OFF_HEALTH_BLOCK_TOTAL_SCALAR] = min(total / TOTAL_MAX, 1.0)
