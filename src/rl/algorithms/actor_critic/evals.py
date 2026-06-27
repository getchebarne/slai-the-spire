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
from src.rl.types import Level
from src.rl.types import NUM_ACTION_TYPES


# Behaviorally-loaded ActionTypes surfaced to TensorBoard, keyed by int(ActionType). Resolved
# against the engine enum at import, so a rename/removal in slai fails loudly here instead of
# silently mislabeling a tag.
_TELEMETRY_ACTION_NAMES = (
    "CardPurge",
    "ShopPurge",  # P4 deck thinning
    "ShopBuyCard",
    "ShopBuyRelic",
    "ShopBuyPotion",  # P3 shop spend
    "RewardTakeGold",
    "RewardTakePotion",
    "RewardTakeCard",
    "RewardTakeRelic",  # P2 loot
    "PotionUse",
    "PotionDiscard",  # P2 potions
    "Rest",
    "CardUpgrade",  # P1/P3 rest vs upgrade
    "RoomSelect",
    "TurnEnd",  # P3 routing / P5 turn end
)
_TELEMETRY_AT_IDX = {n: int(getattr(slai.ActionType, n)) for n in _TELEMETRY_ACTION_NAMES}
_REWARD_TAKE_IDX = [
    int(getattr(slai.ActionType, n))
    for n in ("RewardTakeGold", "RewardTakePotion", "RewardTakeCard", "RewardTakeRelic")
]
_ROOM_EXIT_IDX = int(slai.ActionType.RoomExit)

# Greedy deterministic play can loop; cap so a hung eval can't wedge its worker
_EVAL_MAX_STEPS = 1000
_EVAL_NUM_EPISODES = 64
_EVAL_SEEDS = tuple(range(_EVAL_NUM_EPISODES))  # fixed test set => low-variance trend


def run_eval_battery(model: ActorCritic, device: torch.device, gamma: float) -> dict[str, float]:
    """Greedy fixed-seed battery returning outcome + behavioral metrics, on the same
    encode -> mask -> forward path as training. Fixed seeds make it a paired test set across
    checkpoints (std/sqrt(N) variance, not the old 1/sqrt(1))."""
    rewards: list[float] = []
    lengths: list[int] = []
    wins: list[bool] = []
    floors: list[int] = []
    t_chosen = torch.zeros(NUM_ACTION_TYPES)
    t_available = torch.zeros(NUM_ACTION_TYPES)
    n_reward = 0
    n_reward_skip = 0
    with torch.no_grad():
        for seed in _EVAL_SEEDS:
            env = slai.GameEnv(ascension=ASCENSION_LEVEL, fast_mode=FAST_MODE)
            obs = env.reset(seed=seed)
            total_reward = 0.0
            length = 0
            terminated = False
            while not terminated and length < _EVAL_MAX_STEPS:
                legal = env.get_legal_actions()
                if not legal:
                    break

                # Greedy step: encode -> mask -> forward
                t_game_state = encode_batch_game_state([obs], device)
                t_mask = build_masks([obs], [legal], device)
                t_action = model(t_game_state, t_mask, greedy=True)
                action_type = int(t_action.idx[0, Level.ACTION_TYPE].item())

                # Tally per-ActionType chosen vs available, plus reward-room skips
                t_mask_at = t_mask.mask_action_type[0]
                t_chosen[action_type] += 1.0
                t_available += t_mask_at.float()
                if t_mask_at[_REWARD_TAKE_IDX].any():
                    n_reward += 1
                    if action_type == _ROOM_EXIT_IDX:
                        n_reward_skip += 1

                prev = obs
                obs, terminated = env.step(get_action(t_action, 0))
                total_reward += float(compute_reward(prev, obs, terminated, gamma).sum())
                length += 1
            rewards.append(total_reward)
            lengths.append(length)
            wins.append(bool(terminated and obs.character.health > 0))
            floors.append(obs.map.y_current or 0)

    # Outcome metrics
    n = len(rewards)
    metrics: dict[str, float] = {
        "Eval/win_rate": sum(wins) / n,
        "Eval/avg_floor": sum(floors) / n,
        "Eval/avg_length": sum(lengths) / n,
        "Eval/reward_mean": float(np.mean(rewards)),
        "Eval/reward_std": float(np.std(rewards)),
    }
    # Per-ActionType conditional choice rate (chosen when available)
    for name, idx in _TELEMETRY_AT_IDX.items():
        avail = t_available[idx].item()
        metrics[f"Eval/cond_rate/{name}"] = t_chosen[idx].item() / avail if avail > 0 else 0.0
    if n_reward > 0:
        metrics["Eval/reward_skip_rate"] = n_reward_skip / n_reward
    return metrics


def eval_battery_worker(conn: Connection, model_config: dict, gamma: float) -> None:
    """Persistent eval worker (mirrors `_rollout_worker`'s lifecycle): receive (weights,
    iteration), run the greedy battery, reply with metrics. `None` stops it. Single-threaded,
    so it rides the spare core off the training critical path."""
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
