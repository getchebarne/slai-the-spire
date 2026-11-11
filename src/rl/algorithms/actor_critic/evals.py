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
from src.rl.algorithms.actor_critic.curriculum import ascensions
from src.rl.algorithms.actor_critic.envs import EpisodeResult
from src.rl.algorithms.actor_critic.envs import new_env
from src.rl.algorithms.actor_critic.envs import step_env
from src.rl.batches import allocate_states
from src.rl.batches import map_tensors
from src.rl.batches import read_states
from src.rl.encoding import encode_states_into
from src.rl.models import ActorCritic
from src.rl.utils import parent_alive


_EVAL_NUM_EPISODES = 64
_EVAL_SEEDS = tuple(range(_EVAL_NUM_EPISODES))  # Fixed test set => low-variance trend


def _run_eval_episode(
    model: ActorCritic, gamma: float, ascension: int, seed: int
) -> EpisodeResult:
    env, game_state = new_env(ascension, seed)
    stats = EpisodeResult()
    states = map_tensors(allocate_states(1, 1), lambda x_tensor: x_tensor[0])
    done = False
    while not done:
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            raise RuntimeError(
                f"Non-terminal state has no legal action (seed={seed}, step={stats.length})"
            )

        # Greedy step: Encode (masks included) -> Forward
        encode_states_into(states, [game_state], [legal_actions])
        state_batch = read_states(states)
        policy = model.act(state_batch, greedy=True)

        # Game step
        action = action_from_actiontype(policy, 0, game_state)
        game_state, _, done = step_env(env, game_state, stats, action, gamma)
    return stats


def run_eval_battery(
    model: ActorCritic, gamma: float, ascension_current: float
) -> list[EpisodeResult]:
    """Run the battery at curriculum position ascension_current, `model` on the CPU.

    The games play training's mix of levels at ascension_current: the first
    round(frac(ascension_current)*games) play the frontier level ceil(ascension_current), the rest
    floor(ascension_current).

    Parameters
    ----------
    model : ActorCritic
        The model to evaluate; it plays greedily.
    gamma : float
        The discount the rewards and discounted returns use.
    ascension_current : float
        The curriculum position.

    Returns
    -------
    list[EpisodeResult]
        The battery's finished episodes, in seed order.
    """
    with torch.no_grad():
        episodes = [
            _run_eval_episode(model, gamma, ascension, seed)
            for ascension, seed in zip(
                ascensions(ascension_current, len(_EVAL_SEEDS)), _EVAL_SEEDS
            )
        ]

    return episodes


@dataclass
class Evaluator:
    """The eval process, the parent's end of its pipe, and whether a battery is running."""

    process: BaseProcess
    conn: Connection
    busy: bool


def start_eval(model_config: dict[str, Any], gamma: float) -> Evaluator:
    """Start the eval process, idle until its first weights arrive.

    Parameters
    ----------
    model_config : dict[str, Any]
        ActorCritic's keyword arguments, for the eval process's own model.
    gamma : float
        The discount the batteries' rewards and discounted returns use.

    Returns
    -------
    Evaluator
        The eval process and the parent's end of its pipe, with no battery running.
    """
    ctx = mp.get_context("spawn")
    conn, child = ctx.Pipe()
    process = ctx.Process(target=_eval_loop, args=(child, model_config, gamma), daemon=True)
    process.start()
    child.close()  # The eval process holds its own copy
    return Evaluator(process, conn, busy=False)


def submit_eval(
    evaluator: Evaluator, model: ActorCritic, iteration: int, ascension_current: float
) -> None:
    """Hand the eval process the model's CPU weights for a battery at `ascension_current`.

    It does nothing while a battery is running.

    Parameters
    ----------
    evaluator : Evaluator
        The eval process, marked busy once handed the weights.
    model : ActorCritic
        The model whose weights the battery plays.
    iteration : int
        The iteration the weights are from, which the battery's result carries back.
    ascension_current : float
        The curriculum position the battery plays at.
    """
    _check_alive(evaluator)
    if not evaluator.busy:
        evaluator.conn.send((_cpu_state_dict(model), iteration, ascension_current))
        evaluator.busy = True


def poll_eval(evaluator: Evaluator) -> tuple[int, float, list[EpisodeResult]] | None:
    """Return a finished battery's (iteration, ascension_current, episode_results), or None.

    Parameters
    ----------
    evaluator : Evaluator
        The eval process, marked idle once its battery's result is taken.

    Returns
    -------
    tuple[int, float, list[EpisodeResult]] | None
        The battery's iteration, curriculum position and finished episodes; None while no
        battery has finished.
    """
    _check_alive(evaluator)
    if not evaluator.busy or not evaluator.conn.poll():
        return None
    evaluator.busy = False
    return evaluator.conn.recv()


def stop_evaluator(evaluator: Evaluator) -> None:
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
            f"The eval process died (exitcode {evaluator.process.exitcode}); "
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
    return {name: x_weight.cpu() for name, x_weight in model.state_dict().items()}


def _eval_loop(conn: Connection, model_config: dict[str, Any], gamma: float) -> None:
    # A Ctrl-C or torchrun's SIGTERM reaches the whole process group: ignore them, so the
    # training loop can finish its iteration and checkpoint before it stops this process
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    # A Learner killed outright can't stop this process
    threading.Thread(target=_exit_when_orphaned, daemon=True).start()
    torch.set_num_threads(1)  # Keeps the eval on the spare core, off the training critical path
    model = ActorCritic(**model_config)
    model.eval()
    while True:
        try:
            state_dict, iteration, ascension_current = conn.recv()
        except (EOFError, OSError):  # The Learner is gone
            return
        model.load_state_dict(state_dict)
        episode_results = run_eval_battery(model, gamma, ascension_current)
        try:
            conn.send((iteration, ascension_current, episode_results))
        except BrokenPipeError:  # The Learner is gone
            return


def _exit_when_orphaned() -> None:
    """End this process, even mid-battery, once the Learner is gone."""
    while parent_alive():
        time.sleep(1.0)
    os._exit(0)
