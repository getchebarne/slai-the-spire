import numpy as np
from slai import GameState

from src.rl.schema.reward import OFF_REWARD_CARDS_FORCED
from src.rl.schema.reward import OFF_REWARD_RELICS_EXCLUSIVE


def encode_reward_meta_into(game_state: GameState, out: np.ndarray) -> None:
    """Write reward flags, leaving states without a reward zero."""
    reward = game_state.reward
    if reward is None:
        return
    out[OFF_REWARD_RELICS_EXCLUSIVE] = float(reward.relics_exclusive)
    out[OFF_REWARD_CARDS_FORCED] = float(reward.cards_forced)
