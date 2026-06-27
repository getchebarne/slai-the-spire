from multiprocessing.connection import Connection

import numpy as np
import slai
import torch

from src.rl.constants import ASCENSION_LEVEL
from src.rl.constants import FAST_MODE
from src.rl.encoding.state import encode_batch_game_state
from src.rl.masks import build_masks
from src.rl.models import ActorCritic
from src.rl.models.actor_critic import get_action
from src.rl.reward import compute_reward

# Greedy deterministic play can loop; cap so a hung eval can't wedge its worker
_EVAL_MAX_STEPS = 1000
_EVAL_NUM_EPISODES = 64
_EVAL_SEEDS = tuple(range(_EVAL_NUM_EPISODES))  # fixed test set => low-variance trend


def run_eval_battery(model: ActorCritic, device: torch.device, gamma: float) -> dict[str, float]:
    ep_rewards = []
    ep_lengths = []
    ep_wins = []
    ep_floors = []
    with torch.no_grad():
        for seed in _EVAL_SEEDS:
            env = slai.GameEnv(ascension=ASCENSION_LEVEL, fast_mode=FAST_MODE)
            obs = env.reset(seed=seed)
            reward_total = 0.0
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
                obs, terminated = env.step(get_action(t_action, 0))
                reward_total += float(compute_reward(prev, obs, terminated, gamma).sum())
                length += 1

            ep_rewards.append(reward_total)
            ep_lengths.append(length)
            ep_wins.append(bool(terminated and obs.character.health > 0))
            ep_floors.append(obs.map.y_current or 0)

    # Outcome metrics
    n = len(_EVAL_SEEDS)
    metrics = {
        "evals/win_rate": sum(ep_wins) / n,
        "evals/floor_avg": sum(ep_floors) / n,
        "evals/length_avg": sum(ep_lengths) / n,
        "evals/reward_avg": float(np.mean(ep_rewards)),
        "evals/reward_std": float(np.std(ep_rewards)),
    }
    return metrics


def eval_battery_worker(conn: Connection, model_config: dict, gamma: float) -> None:
    torch.set_num_threads(1)
    device = torch.device("cpu")
    model = ActorCritic(**model_config)
    model.eval()
    while True:
        msg = conn.recv()
        if msg is None:
            return
        state_dict, iteration = msg
        model.load_state_dict(state_dict)
        conn.send((iteration, run_eval_battery(model, device, gamma)))
