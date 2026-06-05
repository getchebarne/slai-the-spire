import numpy as np
import torch
from slai import Reward

from src.rl.constants import GOLD_CAP
from src.rl.constants import MAX_SIZE_COMBAT_CARD_REWARD
from src.rl.encoding.card import ENCODING_DIM_CARD
from src.rl.encoding.card import encode_card_into
from src.rl.encoding.potion import ENCODING_DIM_POTION
from src.rl.encoding.potion import encode_potion_into
from src.rl.encoding.relic import ENCODING_DIM_RELIC
from src.rl.encoding.relic import encode_relic_into
from src.rl.utils import get_sqrt_norm


_GOLD_MAX = 100  # offered gold (monster 10-20, elite 25-44, chest 25-75); sqrt-scaled

_ENCODING_DIM_REWARD_META = (
    1                          # Gold scalar
    + 1                        # Has gold
    + 1                        # Gold after taking (wallet + offered)
    + 1                        # Has relic
    + ENCODING_DIM_RELIC       # Relic features
    + 1                        # Has potion
    + ENCODING_DIM_POTION      # Potion features
)


def _encode_reward_into(reward: Reward, character_gold: int, out: np.ndarray) -> None:
    pos = 0

    # Gold
    gold = reward.gold
    out[pos] = get_sqrt_norm(gold, _GOLD_MAX) if gold is not None else 0.0
    out[pos + 1] = float(gold is not None)
    out[pos + 2] = get_sqrt_norm(character_gold + (gold or 0), GOLD_CAP)
    pos += 3

    # Relic
    out[pos] = float(reward.relic is not None)
    pos += 1
    if reward.relic is not None:
        encode_relic_into(reward.relic, pos, out)
    pos += ENCODING_DIM_RELIC

    # Potion
    out[pos] = float(reward.potion is not None)
    pos += 1
    if reward.potion is not None:
        encode_potion_into(reward.potion, pos, out)
    pos += ENCODING_DIM_POTION


def encode_batch_rewards(
    batch_reward: list[Reward | None], batch_gold: list[int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(batch_reward)

    # Pre-allocate NumPy arrays
    x_cards = np.zeros((batch_size, MAX_SIZE_COMBAT_CARD_REWARD, ENCODING_DIM_CARD), dtype=np.float32)
    x_cards_pad = np.zeros((batch_size, MAX_SIZE_COMBAT_CARD_REWARD), dtype=bool)
    x_meta = np.zeros((batch_size, _ENCODING_DIM_REWARD_META), dtype=np.float32)

    for b, reward in enumerate(batch_reward):
        if reward is None:
            continue

        for i, card in enumerate(reward.cards):
            encode_card_into(card, 0, x_cards[b, i])
            x_cards_pad[b, i] = True

        _encode_reward_into(reward, batch_gold[b], x_meta[b])

    return (
        torch.from_numpy(x_cards).to(device),
        torch.from_numpy(x_cards_pad).to(device),
        torch.from_numpy(x_meta).to(device),
    )
