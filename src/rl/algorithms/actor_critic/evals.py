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
from slai import GameEnv
from slai import GameState

from src.rl.actions import engine_action
from src.rl.algorithms.actor_critic.curriculum import ascensions
from src.rl.algorithms.actor_critic.envs import EpisodeResult
from src.rl.algorithms.actor_critic.envs import new_env
from src.rl.algorithms.actor_critic.envs import step_env
from src.rl.batches import allocate_states
from src.rl.batches import map_tensors
from src.rl.batches import read_states
from src.rl.batches import to_device
from src.rl.encoding import encode_states_into
from src.rl.models import ActorCritic
from src.rl.types import Reserved
from src.rl.types import States
from src.rl.utils import parent_alive


_EVAL_NUM_EPISODES = 256
_EVAL_SEEDS = tuple(range(_EVAL_NUM_EPISODES))

# The share of its GPU's memory the eval process may reserve
_EVAL_MEMORY_FRACTION = 0.1


@dataclass
class EvalBatteryResults:
    """One finished eval battery and how it was played."""

    iteration: int                        # The iteration whose weights it played
    ascension_current: float              # The curriculum position it played at
    episode_results: list[EpisodeResult]  # In seed order
    bytes_reserved_peak: int              # The eval process's peak GPU memory; 0 on the CPU


@dataclass
class Evaluator:
    """The eval process, the parent's end of its pipe, and whether a battery is running."""

    process: BaseProcess
    conn: Connection
    busy: bool


@dataclass
class _EvalConfig:
    """The eval process's config, fixed for the run."""

    model_config: dict[str, Any]
    gamma: float
    device: torch.device

    # The Learner process's torch settings, which a spawned process doesn't inherit
    matmul_precision: str
    deterministic: bool
    use_duck_shape: bool


def _run_eval_battery(
    model: ActorCritic, gamma: float, ascension_current: float, device: torch.device
) -> list[EpisodeResult]:
    """Run the battery at curriculum position ascension_current, every game in lockstep.

    The games play training's mix of levels at ascension_current: the first
    round(frac(ascension_current)*games) play the frontier level ceil(ascension_current), the rest
    floor(ascension_current). Each step runs one greedy forward over the games still running, until
    every game ends or reaches the step limit.

    Parameters
    ----------
    model : ActorCritic
        The model to evaluate, on `device`; it plays greedily.
    gamma : float
        The discount the rewards and discounted returns use.
    ascension_current : float
        The curriculum position.
    device : torch.device
        The model's device.

    Returns
    -------
    list[EpisodeResult]
        The battery's finished episodes, in seed order.
    """
    batch_env: list[GameEnv] = []
    batch_game_state: list[GameState] = []
    for ascension, seed in zip(ascensions(ascension_current, len(_EVAL_SEEDS)), _EVAL_SEEDS):
        env, game_state = new_env(ascension, seed)
        batch_env.append(env)
        batch_game_state.append(game_state)

    episode_results = [EpisodeResult() for _ in _EVAL_SEEDS]

    # The games still running, by index in seed order
    batch_idx_game = list(range(len(_EVAL_SEEDS)))
    states_reserved: States[Reserved] | None = None
    with torch.no_grad():
        while batch_idx_game:
            batch_legal_actions = []
            for idx_game in batch_idx_game:
                legal_actions = batch_env[idx_game].get_legal_actions()
                if not legal_actions:
                    raise RuntimeError(
                        f"Non-terminal state has no legal action (seed={_EVAL_SEEDS[idx_game]}, "
                        f"step={episode_results[idx_game].length})"
                    )

                batch_legal_actions.append(legal_actions)

            # The encoder writes exactly its buffer's number of states: a new buffer as games end
            if states_reserved is None or len(states_reserved.x_context) != len(batch_idx_game):
                states_reserved = map_tensors(
                    allocate_states(1, len(batch_idx_game)), lambda x_tensor: x_tensor[0]
                )

            # Greedy step: Encode (masks included) -> Forward, the running games in one batch
            encode_states_into(
                states_reserved,
                [batch_game_state[idx_game] for idx_game in batch_idx_game],
                batch_legal_actions,
            )
            states = to_device(read_states(states_reserved), device)
            policy = model.action_select(states, greedy=True)

            # Game step: the games that end leave the batch
            batch_done = []
            for idx_game, (idx_action_type, idx_selection, idx_target) in zip(
                batch_idx_game, policy.x_action.tolist()
            ):
                action = engine_action(
                    idx_action_type, idx_selection, idx_target, batch_game_state[idx_game]
                )
                batch_game_state[idx_game], _, _, done = step_env(
                    batch_env[idx_game],
                    batch_game_state[idx_game],
                    episode_results[idx_game],
                    action,
                    gamma,
                )
                batch_done.append(done)
            batch_idx_game = [
                idx_game for idx_game, done in zip(batch_idx_game, batch_done) if not done
            ]

    return episode_results


def start_eval(model_config: dict[str, Any], gamma: float, device: torch.device) -> Evaluator:
    """Start the eval process, idle until its first weights arrive.

    It plays on `device` with the Learner's torch settings, as the Actor does.

    Parameters
    ----------
    model_config : dict[str, Any]
        ActorCritic's keyword arguments, for the eval process's own model.
    gamma : float
        The discount the batteries' rewards and discounted returns use.
    device : torch.device
        The device the eval process plays on.

    Returns
    -------
    Evaluator
        The eval process and the parent's end of its pipe, with no battery running.
    """
    eval_config = _EvalConfig(
        model_config=model_config,
        gamma=gamma,
        device=device,
        matmul_precision=torch.get_float32_matmul_precision(),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        use_duck_shape=torch.fx.experimental._config.use_duck_shape,
    )
    ctx = mp.get_context("spawn")
    conn, child = ctx.Pipe()
    process = ctx.Process(target=_eval_loop, args=(child, eval_config), daemon=True)
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


def poll_eval(evaluator: Evaluator) -> EvalBatteryResults | None:
    """Return a finished battery, or None.

    Parameters
    ----------
    evaluator : Evaluator
        The eval process, marked idle once its battery's result is taken.

    Returns
    -------
    EvalBatteryResults | None
        The finished battery; None while no battery has finished.
    """
    _check_alive(evaluator)
    if not evaluator.busy or not evaluator.conn.poll():
        return None
    
    evaluator.busy = False

    return evaluator.conn.recv()


def stop_evaluator(evaluator: Evaluator) -> None:
    """Kill the eval process, mid-battery or not, and wait for it to exit.

    Parameters
    ----------
    evaluator : Evaluator
        The eval process to stop.
    """
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


def _eval_loop(conn: Connection, eval_config: _EvalConfig) -> None:
    """Play a battery on each set of weights that arrives and send back its result.

    It ends once the Learner is gone.

    Parameters
    ----------
    conn : Connection
        The eval process's end of the pipe: the weights, their iteration and the curriculum
        position come in, the battery's result goes out.
    eval_config : _EvalConfig
        The eval process's config, fixed for the run.
    """

    # Wait for the Learner to finish the current iteration and save the checkpoint before exit
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    # A Learner killed outright can't stop this process
    threading.Thread(target=_exit_when_orphaned, daemon=True).start()
    torch.set_num_threads(1)  # Keeps the eval on the spare core, off the training critical path

    # Set the eval's device and cap its share of the device's memory
    device = eval_config.device
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(_EVAL_MEMORY_FRACTION, device)

    # Set PyTorch config, as the Actor's
    torch.set_float32_matmul_precision(eval_config.matmul_precision)
    torch.use_deterministic_algorithms(eval_config.deterministic)
    torch.fx.experimental._config.use_duck_shape = eval_config.use_duck_shape

    model = ActorCritic(**eval_config.model_config).to(device)
    model.eval()
    while True:
        try:
            state_dict, iteration, ascension_current = conn.recv()
        except (EOFError, OSError):
            # The Learner is gone
            return
        
        model.load_state_dict(state_dict)
        episode_results = _run_eval_battery(model, eval_config.gamma, ascension_current, device)
        bytes_reserved_peak = 0
        if device.type == "cuda":
            bytes_reserved_peak = torch.cuda.max_memory_reserved(device)

        try:
            conn.send(EvalBatteryResults(iteration, ascension_current, episode_results, bytes_reserved_peak))
        except BrokenPipeError:
            # The Learner is gone
            return


def _exit_when_orphaned() -> None:
    """End this process, even mid-battery, once the Learner is gone."""
    while parent_alive():
        time.sleep(1.0)

    os._exit(0)
