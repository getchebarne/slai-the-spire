from enum import StrEnum
from enum import auto

import numpy as np
import torch
from slai import GameState
from slai import Reward

from src.rl.types import make_channel_allocator


class RewardMetaChannel(StrEnum):
    RELICS_EXCLUSIVE = auto()  # Boss rewards: take one Relic, lose the rest
    CARDS_FORCED = auto()  # The Library: one selection, no skip


CHANNEL_LAYOUT_REWARD_META, _alloc = make_channel_allocator()
OFF_REWARD_RELICS_EXCLUSIVE = _alloc(RewardMetaChannel.RELICS_EXCLUSIVE)
OFF_REWARD_CARDS_FORCED = _alloc(RewardMetaChannel.CARDS_FORCED)

ENCODING_DIM_REWARD_META = sum(w for _, w in CHANNEL_LAYOUT_REWARD_META)


def _encode_reward_meta_into(reward: Reward, out: np.ndarray) -> None:
    out[OFF_REWARD_RELICS_EXCLUSIVE] = float(reward.relics_exclusive)
    out[OFF_REWARD_CARDS_FORCED] = float(reward.cards_forced)


def encode_batch_rewards(game_states: list[GameState], device: torch.device) -> torch.Tensor:
    """Reward meta only — the offered card/relic/potion entities are encoded by
    their entity-class encoders (reward slices per the encoders)."""
    batch_size = len(game_states)
    np_meta = np.zeros((batch_size, ENCODING_DIM_REWARD_META), dtype=np.float32)

    for b, game_state in enumerate(game_states):
        reward = game_state.reward
        if reward is None:
            continue

        _encode_reward_meta_into(reward, np_meta[b])

    return torch.from_numpy(np_meta).to(device)
