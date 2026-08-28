from enum import StrEnum
from enum import auto

import numpy as np
import torch
from slai import GameEnv
from slai import Reward

from src.rl.constants import GOLD_CAP
from src.rl.types import make_channel_allocator
from src.rl.utils import get_sqrt_norm


# Widest reward-gold offer (boss roll + Golden Idol); sqrt-scaled
_GOLD_MAX = GameEnv.REWARD_GOLD_MAX


class RewardMetaChannel(StrEnum):
    GOLD = auto()
    GOLD_AFTER_TAKING = auto()
    RELICS_EXCLUSIVE = auto()  # boss rewards: take one relic, lose the rest


CHANNEL_LAYOUT_REWARD_META, _alloc = make_channel_allocator()
OFF_REWARD_GOLD = _alloc(RewardMetaChannel.GOLD)
OFF_REWARD_GOLD_AFTER_TAKING = _alloc(RewardMetaChannel.GOLD_AFTER_TAKING)
OFF_REWARD_RELICS_EXCLUSIVE = _alloc(RewardMetaChannel.RELICS_EXCLUSIVE)

ENCODING_DIM_REWARD_META = sum(w for _, w in CHANNEL_LAYOUT_REWARD_META)


def _encode_reward_meta_into(reward: Reward, character_gold: int, out: np.ndarray) -> None:
    out[OFF_REWARD_GOLD] = (
        get_sqrt_norm(reward.gold, _GOLD_MAX) if reward.gold is not None else 0.0
    )
    out[OFF_REWARD_GOLD_AFTER_TAKING] = get_sqrt_norm(
        character_gold + (reward.gold or 0), GOLD_CAP
    )
    out[OFF_REWARD_RELICS_EXCLUSIVE] = float(reward.relics_exclusive)


def encode_batch_rewards(
    batch_reward: list[Reward | None],
    batch_gold: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Reward meta only — the offered card/relic/potion entities are encoded by
    their entity-class encoders (reward slices per the encoders)."""
    batch_size = len(batch_reward)
    np_meta = np.zeros((batch_size, ENCODING_DIM_REWARD_META), dtype=np.float32)

    for b, reward in enumerate(batch_reward):
        if reward is None:
            continue

        _encode_reward_meta_into(reward, batch_gold[b], np_meta[b])

    return torch.from_numpy(np_meta).to(device)
