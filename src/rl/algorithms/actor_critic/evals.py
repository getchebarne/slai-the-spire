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
from src.rl.encoding.state import encode_batch_game_state
from src.rl.masks import build_masks
from src.rl.models import ActorCritic
from src.rl.reward import REWARD_STREAMS
from src.rl.reward import compute_reward
from src.rl.reward import cumulative_floor
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
            env = slai.GameEnv(ascension=ascension, fast_mode=FAST_MODE)
            obs = env.reset(seed=seed)
            stream_rewards = np.zeros(len(REWARD_STREAMS))
            length = 0
            terminated = False
            while not terminated and length < _EVAL_MAX_STEPS:
                legal = env.get_legal_actions()
                if not legal:
                    raise RuntimeError(
                        f"non-terminal state has no legal action (seed={seed}, step={length})"
                    )

                # Greedy step: Encode -> Mask -> Forward
                t_game_state = encode_batch_game_state([obs], device)
                t_mask = build_masks([obs], [legal], device)
                t_action, _ = model(t_game_state, t_mask, greedy=True)

                # Game step
                prev = obs
                obs, terminated = env.step(action_from_actiontype(t_action, 0, obs))
                stream_rewards += compute_reward(prev, obs, terminated, gamma)
                length += 1

            episodes.append(
                EpisodeStats(
                    stream_rewards=stream_rewards,
                    length=length,
                    won=bool(terminated and obs.character.health > 0),
                    floor=cumulative_floor(obs),
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
