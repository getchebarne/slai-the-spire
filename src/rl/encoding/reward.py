import numpy as np
import torch
from slai import Reward

from src.rl.constants import GOLD_CAP
from src.rl.constants import MAX_POTION_REWARDS
from src.rl.constants import MAX_RELIC_REWARDS
from src.rl.constants import MAX_SIZE_COMBAT_CARD_REWARD
from src.rl.encoding.card import ENCODING_DIM_CARD
from src.rl.encoding.card import encode_card_row
from src.rl.encoding.potion import ENCODING_DIM_POTION
from src.rl.encoding.potion import encode_potion_into
from src.rl.encoding.relic import ENCODING_DIM_RELIC
from src.rl.encoding.relic import encode_relic_into
from src.rl.utils import get_sqrt_norm


_GOLD_MAX = 100  # offered gold (monster 10-20, elite 25-44, chest 25-75); sqrt-scaled

# Offered relic/potion are encoded as their own pure entity tensors (projected via
# the shared relic/potion projections); only the gold scalars live in meta.
_ENCODING_DIM_REWARD_META = (
    1                          # Gold scalar
    + 1                        # Has gold
    + 1                        # Gold after taking (wallet + offered)
)


def _encode_reward_meta_into(reward: Reward, character_gold: int, out: np.ndarray) -> None:
    gold = reward.gold
    out[0] = get_sqrt_norm(gold, _GOLD_MAX) if gold is not None else 0.0
    out[1] = float(gold is not None)
    out[2] = get_sqrt_norm(character_gold + (gold or 0), GOLD_CAP)


def encode_batch_rewards(
    batch_reward: list[Reward | None],
    batch_gold: list[int],
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    batch_size = len(batch_reward)

    # Pre-allocate NumPy arrays
    x_cards = np.zeros((batch_size, MAX_SIZE_COMBAT_CARD_REWARD, ENCODING_DIM_CARD), dtype=np.float32)
    x_cards_pad = np.zeros((batch_size, MAX_SIZE_COMBAT_CARD_REWARD), dtype=bool)
    x_relic = np.zeros((batch_size, MAX_RELIC_REWARDS, ENCODING_DIM_RELIC), dtype=np.float32)
    x_relic_pad = np.zeros((batch_size, MAX_RELIC_REWARDS), dtype=bool)
    x_potion = np.zeros((batch_size, MAX_POTION_REWARDS, ENCODING_DIM_POTION), dtype=np.float32)
    x_potion_pad = np.zeros((batch_size, MAX_POTION_REWARDS), dtype=bool)
    x_meta = np.zeros((batch_size, _ENCODING_DIM_REWARD_META), dtype=np.float32)

    for b, reward in enumerate(batch_reward):
        if reward is None:
            continue

        # Reward cards aren't played from here; energy_current is irrelevant -> 0
        for i, card in enumerate(reward.cards):
            encode_card_row(card, 0, x_cards[b, i])
            x_cards_pad[b, i] = True

        if reward.relic is not None:
            encode_relic_into(reward.relic, 0, x_relic[b, 0])
            x_relic_pad[b, 0] = True

        if reward.potion is not None:
            encode_potion_into(reward.potion, 0, x_potion[b, 0])
            x_potion_pad[b, 0] = True

        _encode_reward_meta_into(reward, batch_gold[b], x_meta[b])

    return (
        torch.from_numpy(x_cards).to(device),
        torch.from_numpy(x_cards_pad).to(device),
        torch.from_numpy(x_relic).to(device),
        torch.from_numpy(x_relic_pad).to(device),
        torch.from_numpy(x_potion).to(device),
        torch.from_numpy(x_potion_pad).to(device),
        torch.from_numpy(x_meta).to(device),
    )
