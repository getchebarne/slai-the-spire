import numpy as np
from slai import GameState

from src.rl.encoding.layout import make_channel_allocator


_CHANNEL_LAYOUT_REWARD_META, _alloc = make_channel_allocator()
_OFF_REWARD_RELICS_EXCLUSIVE = _alloc(
    "relics_exclusive"
)  # Boss Rewards: take one Relic, lose the rest
_OFF_REWARD_CARDS_FORCED = _alloc("cards_forced")  # The Library: one selection, no skip

ENCODING_DIM_REWARD_META = sum(w for _, w in _CHANNEL_LAYOUT_REWARD_META)


def encode_reward_meta_into(game_state: GameState, out: np.ndarray) -> None:
    """Write reward flags, leaving states without a reward zero."""
    reward = game_state.reward
    if reward is None:
        return
    out[_OFF_REWARD_RELICS_EXCLUSIVE] = float(reward.relics_exclusive)
    out[_OFF_REWARD_CARDS_FORCED] = float(reward.cards_forced)
