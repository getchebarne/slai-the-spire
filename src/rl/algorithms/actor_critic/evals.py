from multiprocessing.connection import Connection

import numpy as np
import slai
import torch

from src.rl.algorithms.actor_critic.episode import KEY_ACT2_RATE
from src.rl.algorithms.actor_critic.episode import KEY_FLOOR_AVG
from src.rl.algorithms.actor_critic.episode import KEY_REWARD_AVG
from src.rl.algorithms.actor_critic.episode import KEY_REWARD_STD
from src.rl.algorithms.actor_critic.episode import KEY_WIN_RATE
from src.rl.algorithms.actor_critic.episode import PREFIX_EVAL
from src.rl.algorithms.actor_critic.episode import EpisodeStats
from src.rl.algorithms.actor_critic.episode import aggregate_episodes
from src.rl.constants import FAST_MODE
from src.rl.constants import NEOW
from src.rl.encoding.state import encode_batch_game_state
from src.rl.masks import build_masks
from src.rl.models import ActorCritic
from src.rl.reward import REWARD_STREAMS
from src.rl.reward import compute_reward
from src.rl.reward import cumulative_floor
from src.rl.reward import shop_gold_spent
from src.rl.reward import wasted_potion_discard
from src.rl.utils import action_from_actiontype


# Greedy deterministic play can loop; cap so a hung eval can't wedge its worker.
# Two-act episodes run ~350-450 steps; 2000 ≈ 4.5x nominal — a lower cap would
# silently record a truncated run as a loss and corrupt the curriculum gate.
_EVAL_MAX_STEPS = 2000
_EVAL_NUM_EPISODES = 64
_EVAL_SEEDS = tuple(range(_EVAL_NUM_EPISODES))  # fixed test set => low-variance trend

# Full eval tags, composed from the shared prefix + key constants; imported by
# train()'s eval print.
EVAL_WIN_RATE = PREFIX_EVAL + KEY_WIN_RATE
EVAL_FLOOR_AVG = PREFIX_EVAL + KEY_FLOOR_AVG
EVAL_REWARD_AVG = PREFIX_EVAL + KEY_REWARD_AVG
EVAL_ACT2_RATE = PREFIX_EVAL + KEY_ACT2_RATE
EVAL_REWARD_STD = PREFIX_EVAL + KEY_REWARD_STD


def run_eval_battery(
    model: ActorCritic, device: torch.device, gamma: float, ascension: int
) -> dict[str, float]:
    episodes: list[EpisodeStats] = []
    with torch.no_grad():
        for seed in _EVAL_SEEDS:
            env = slai.GameEnv(ascension=ascension, fast_mode=FAST_MODE, neow=NEOW)
            obs = env.reset(seed=seed)
            stream_rewards = np.zeros(len(REWARD_STREAMS))
            stream_returns = np.zeros(len(REWARD_STREAMS))
            length = 0
            gold_gained = 0
            gold_spent = 0
            potion_wastes = 0
            cards_added = 0
            purges = 0
            terminated = False
            while not terminated and length < _EVAL_MAX_STEPS:
                legal = env.get_legal_actions()
                if not legal:
                    raise RuntimeError(
                        f"non-terminal state has no legal action (seed={seed}, step={length})"
                    )

                # Greedy step: Encode -> Mask -> Forward
                x_game_state = encode_batch_game_state([obs], device)
                x_mask = build_masks([obs], [legal], x_game_state, device)
                x_action, _ = model(x_game_state, x_mask, greedy=True)

                # Game step (a room pick decodes through the rooms row)
                prev = obs
                action = action_from_actiontype(
                    x_action, 0, x_game_state.room_idxs[0].cpu().numpy(), obs
                )
                obs, terminated = env.step(action)
                reward = compute_reward(prev, obs, terminated, gamma, action)
                stream_rewards += reward
                stream_returns += gamma**length * reward
                gold_gained += max(0, obs.character.gold - prev.character.gold)
                gold_spent += shop_gold_spent(prev, obs, action)
                potion_wastes += wasted_potion_discard(prev, action)
                cards_added += max(0, len(obs.card_pile_deck) - len(prev.card_pile_deck))
                purges += action.action_type == slai.ActionType.ShopPurge
                length += 1

            episodes.append(
                EpisodeStats(
                    stream_rewards=stream_rewards,
                    length=length,
                    won=bool(terminated and obs.character.health > 0),
                    floor=cumulative_floor(obs),
                    stream_returns=stream_returns,
                    gold_gained=gold_gained,
                    gold_spent=gold_spent,
                    potion_wastes=potion_wastes,
                    deck_size=len(obs.card_pile_deck),
                    cards_added=cards_added,
                    purges=purges,
                )
            )

    return {PREFIX_EVAL + key: value for key, value in aggregate_episodes(episodes).items()}


def eval_battery_worker(conn: Connection, model_config: dict, gamma: float) -> None:
    torch.set_num_threads(1)
    device = torch.device("cpu")
    model = ActorCritic(**model_config)
    model.eval()
    while True:
        msg = conn.recv()
        if msg is None:
            return
        state_dict, iteration, ascension = msg
        model.load_state_dict(state_dict)
        conn.send((iteration, ascension, run_eval_battery(model, device, gamma, ascension)))
