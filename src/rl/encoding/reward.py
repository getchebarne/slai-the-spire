import numpy as np
import torch
from slai import Reward

from src.rl.constants import GOLD_CAP
from src.rl.utils import get_sqrt_norm

_GOLD_MAX = 100  # offered gold (monster 10-20, elite 25-44, chest 25-75); sqrt-scaled
_ENCODING_DIM_REWARD_META = 1 + 1  # Gold scalar  # Gold after taking (wallet + offered)


def _encode_reward_meta_into(reward: Reward, character_gold: int, out: np.ndarray) -> None:
    out[0] = get_sqrt_norm(reward.gold, _GOLD_MAX) if reward.gold is not None else 0.0
    out[1] = get_sqrt_norm(character_gold + (reward.gold or 0), GOLD_CAP)


def encode_batch_rewards(
    batch_reward: list[Reward | None],
    batch_gold: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Reward meta only — the offered card/relic/potion entities are encoded by
    their entity-class encoders (reward segments per src.rl.index)."""
    batch_size = len(batch_reward)

    x_meta = np.zeros((batch_size, _ENCODING_DIM_REWARD_META), dtype=np.float32)

    for b, reward in enumerate(batch_reward):
        if reward is None:
            continue

        _encode_reward_meta_into(reward, batch_gold[b], x_meta[b])

    return torch.from_numpy(x_meta).to(device)
