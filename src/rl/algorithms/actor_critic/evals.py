import os
import signal
import threading
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any

import torch
import torch.multiprocessing as mp

from src.rl.actions import action_from_actiontype
from src.rl.algorithms.actor_critic.curriculum import env_level
from src.rl.algorithms.actor_critic.episode import KEY_ACT2_RATE
from src.rl.algorithms.actor_critic.episode import KEY_FLOOR_AVG
from src.rl.algorithms.actor_critic.episode import KEY_REWARD_AVG
from src.rl.algorithms.actor_critic.episode import KEY_WIN_RATE
from src.rl.algorithms.actor_critic.episode import PREFIX_EVAL
from src.rl.algorithms.actor_critic.episode import EpisodeStats
from src.rl.algorithms.actor_critic.episode import aggregate_episode_stats
from src.rl.algorithms.actor_critic.episode import finish_episode
from src.rl.algorithms.actor_critic.episode import make_episode_stats
from src.rl.algorithms.actor_critic.episode import tally_step
from src.rl.encoding.state import encode_batch_game_state
from src.rl.models import ActorCritic
from src.rl.reward import compute_reward
from src.rl.utils import new_env
from src.rl.utils import parent_alive
from src.rl.utils import step_scripted


# Greedy deterministic play can loop; cap so a hung eval can't wedge its worker. Keep it
# well above an episode's length: a truncated run counts as a loss for the curriculum gate
_EVAL_MAX_STEPS = 2000
_EVAL_NUM_EPISODES = 64
_EVAL_SEEDS = tuple(range(_EVAL_NUM_EPISODES))  # Fixed test set => low-variance trend

# Full eval tags, composed from the shared prefix + key constants; imported by
# train()'s eval print.
EVAL_WIN_RATE = PREFIX_EVAL + KEY_WIN_RATE
EVAL_FLOOR_AVG = PREFIX_EVAL + KEY_FLOOR_AVG
EVAL_REWARD_AVG = PREFIX_EVAL + KEY_REWARD_AVG
EVAL_ACT2_RATE = PREFIX_EVAL + KEY_ACT2_RATE


def _run_eval_episode(model: ActorCritic, gamma: float, ascension: int, seed: int) -> EpisodeStats:
    env = new_env(ascension)
    obs = env.reset(seed=seed)
    stats = make_episode_stats()
    terminated = False
    while not terminated and stats.length < _EVAL_MAX_STEPS:
        legal = env.get_legal_actions()
        if not legal:
            raise RuntimeError(
                f"non-terminal state has no legal action (seed={seed}, step={stats.length})"
            )

        # Greedy step: Encode (masks included) -> Forward
        x_gs = encode_batch_game_state([obs], [legal], torch.device("cpu"))
        x_action = model.act(x_gs, greedy=True)

        # Game step
        prev = obs
        action = action_from_actiontype(x_action, 0, obs)
        obs, terminated = step_scripted(env, action)
        reward = compute_reward(prev, obs, terminated, gamma, action)
        tally_step(stats, prev, obs, action, reward, gamma)

    finish_episode(stats, obs, terminated)
    return stats


def run_eval_battery(
    model: ActorCritic, gamma: float, c: float, ascension_max: int
) -> dict[str, float]:
    """Run the battery at curriculum position c, `model` on the CPU.

    The games play training's mix of levels at c: the first round(frac(c)*games) play the
    frontier level ceil(c), the rest floor(c).

    Parameters
    ----------
    model : ActorCritic
        The model to evaluate; it plays greedily.
    gamma : float
        The discount the rewards and discounted returns use.
    c : float
        The curriculum position.
    ascension_max : int
        The highest level, which caps every game's.

    Returns
    -------
    dict[str, float]
        The battery's outcome metrics, keyed by their eval/* names.
    """
    with torch.no_grad():
        episodes = [
            _run_eval_episode(model, gamma, env_level(c, i, len(_EVAL_SEEDS), ascension_max), seed)
            for i, seed in enumerate(_EVAL_SEEDS)
        ]

    return {PREFIX_EVAL + key: value for key, value in aggregate_episode_stats(episodes).items()}


@dataclass
class Evaluator:
    """The eval process, the parent's end of its pipe, and whether a battery is running."""

    process: BaseProcess
    conn: Connection
    busy: bool


def start_eval(model_config: dict[str, Any], gamma: float, ascension_max: int) -> Evaluator:
    """Start the eval process, idle until its first weights arrive.

    Parameters
    ----------
    model_config : dict[str, Any]
        ActorCritic's keyword arguments, for the eval process's own model.
    gamma : float
        The discount the batteries' rewards and discounted returns use.
    ascension_max : int
        The highest level, which caps every game's.

    Returns
    -------
    Evaluator
        The eval process and the parent's end of its pipe, with no battery running.
    """
    ctx = mp.get_context("spawn")
    conn, child = ctx.Pipe()
    process = ctx.Process(
        target=_eval_loop, args=(child, model_config, gamma, ascension_max), daemon=True
    )
    process.start()
    child.close()  # The eval process holds its own copy
    return Evaluator(process, conn, busy=False)


def submit_eval(evaluator: Evaluator, model: ActorCritic, iteration: int, c: float) -> None:
    """Hand the eval process the model's CPU weights for a battery at curriculum position `c`.

    It does nothing while a battery is running.

    Parameters
    ----------
    evaluator : Evaluator
        The eval process, marked busy once handed the weights.
    model : ActorCritic
        The model whose weights the battery plays.
    iteration : int
        The iteration the weights are from, which the battery's result carries back.
    c : float
        The curriculum position the battery plays at.
    """
    _check_alive(evaluator)
    if not evaluator.busy:
        evaluator.conn.send((_cpu_state_dict(model), iteration, c))
        evaluator.busy = True


def poll_eval(evaluator: Evaluator) -> tuple[int, float, dict[str, float]] | None:
    """Return a finished battery's (iteration, c, metrics), or None.

    Parameters
    ----------
    evaluator : Evaluator
        The eval process, marked idle once its battery's result is taken.

    Returns
    -------
    tuple[int, float, dict[str, float]] | None
        The battery's iteration, curriculum position and eval/* metrics; None while no battery
        has finished.
    """
    _check_alive(evaluator)
    if not evaluator.busy or not evaluator.conn.poll():
        return None
    evaluator.busy = False
    return evaluator.conn.recv()


def stop_eval(evaluator: Evaluator) -> None:
    evaluator.process.kill()  # It ignores SIGTERM, and an eval in progress has nothing to save
    evaluator.process.join()


def _check_alive(evaluator: Evaluator) -> None:
    """Fail the run if the eval process died: it would never finish a battery.

    Parameters
    ----------
    evaluator : Evaluator
        The eval process to check.
    """
    if not evaluator.process.is_alive():
        raise RuntimeError(
            f"the eval process died (exitcode {evaluator.process.exitcode}); "
            "see its traceback above"
        )


def _cpu_state_dict(model: ActorCritic) -> dict[str, torch.Tensor]:
    """Return the model's weights as CPU tensors, picklable to the eval process on any backend.

    MPS tensors can't be shared across processes; the GPU->CPU copy is negligible.

    Parameters
    ----------
    model : ActorCritic
        The model, on any device.

    Returns
    -------
    dict[str, torch.Tensor]
        The model's state dict, every tensor on the CPU.
    """
    return {k: v.cpu() for k, v in model.state_dict().items()}


def _eval_loop(
    conn: Connection, model_config: dict[str, Any], gamma: float, ascension_max: int
) -> None:
    # A Ctrl-C or torchrun's SIGTERM reaches the whole process group: ignore them, so the
    # training loop can finish its iteration and checkpoint before it stops this process
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    # A learner killed outright can't stop this process
    threading.Thread(target=_exit_when_orphaned, daemon=True).start()
    torch.set_num_threads(1)  # Keeps the eval on the spare core, off the training critical path
    model = ActorCritic(**model_config)
    model.eval()
    while True:
        try:
            state_dict, iteration, c = conn.recv()
        except (EOFError, OSError):  # The learner is gone
            return
        model.load_state_dict(state_dict)
        metrics = run_eval_battery(model, gamma, c, ascension_max)
        try:
            conn.send((iteration, c, metrics))
        except BrokenPipeError:  # The learner is gone
            return


def _exit_when_orphaned() -> None:
    """End this process, even mid-battery, once the learner is gone."""
    while parent_alive():
        time.sleep(1.0)
    os._exit(0)
