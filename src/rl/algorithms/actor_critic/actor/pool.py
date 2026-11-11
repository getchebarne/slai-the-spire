"""The Actor process and the Learner's handle on it.

Version v, the weights after v updates, plays rollout v+1.
"""

import itertools
import os
import random
import signal
import time
from contextlib import ExitStack
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Semaphore
from typing import Any

import torch
import torch.multiprocessing as mp

from src.rl.algorithms.actor_critic.actor.storage import _WeightBuffer
from src.rl.algorithms.actor_critic.actor.storage import allocate_buffer
from src.rl.algorithms.actor_critic.actor.storage import write_weights
from src.rl.algorithms.actor_critic.envs import EpisodeResult
from src.rl.algorithms.actor_critic.workers import Workers
from src.rl.algorithms.actor_critic.workers import snapshot_to_device
from src.rl.algorithms.actor_critic.workers import start_workers
from src.rl.algorithms.actor_critic.workers import step_workers
from src.rl.algorithms.actor_critic.workers import stop_workers
from src.rl.algorithms.actor_critic.workers import write_ascensions
from src.rl.batches import cat_states
from src.rl.models import ActorCritic
from src.rl.types import Rollout
from src.rl.types import StateBatch
from src.rl.utils import parent_alive


_SPAWN = "spawn"


@dataclass
class RolloutConfig:
    num_envs: int
    num_workers: int
    rollout_length: int


@dataclass
class Delivery:
    """One collected rollout and how it was made."""

    rollout: Rollout
    episode_results: list[EpisodeResult]
    seconds_collect: float
    seconds_wait_actor: float  # The Actor's idle time before it: waiting for weights or the Learner
    bytes_reserved_peak: int  # The Actor's peak GPU memory; 0 on the CPU


@dataclass
class ActorHandle:
    """The Learner's handle on its Actor.

    Pass it to `publish`, `receive_rollout` and `stop_actor`. The Actor package owns its
    process, pipe, stop flag and shared weight storage.
    """

    _process: BaseProcess
    _conn_read: Connection  # FIFO; holds at most two Deliveries
    _x_stop: torch.Tensor  # () bool in shared memory, lock-free like the workers' stop flag
    _weight_buffer: _WeightBuffer


@dataclass
class _ActorConfig:
    """The Actor process's config, fixed for the run."""

    model_config: dict[str, Any]
    rollout_config: RolloutConfig
    device: torch.device
    gamma: float

    # The Learner process's state, which a spawned process doesn't inherit
    seed: int  # The workers' seed, drawn from the Learner's stream
    x_rng: torch.Tensor  # The Learner's global generator state on the device
    matmul_precision: str
    num_threads: int
    deterministic: bool
    use_duck_shape: bool


def start_actor(
    model: ActorCritic,
    model_config: dict[str, Any],
    rollout_config: RolloutConfig,
    device: torch.device,
    gamma: float,
) -> ActorHandle:
    """Start the Actor process and allocate the weights the Learner will publish to it.

    The Actor process starts its workers and collects on a model of its own, from the
    Learner's random stream and torch settings. It waits for the first call to `publish`.

    Parameters
    ----------
    model : ActorCritic
        The Learner's model, whose state dictionary sizes the private weight storage.
    model_config : dict[str, Any]
        The model's architecture, which the Actor process builds a model of its own from.
    rollout_config : RolloutConfig
        The games, worker processes and steps a rollout plays.
    device : torch.device
        The device the Actor process plays on.
    gamma : float
        The discount the games' rewards and discounted returns use.

    Returns
    -------
    ActorHandle
        The Learner's handle on the Actor process.
    """
    weight_buffer = allocate_buffer(model.state_dict())

    # Instance the Actor's config. Read by the Actor loop
    actor_config = _ActorConfig(
        model_config=model_config,
        rollout_config=rollout_config,
        device=device,
        gamma=gamma,
        seed=random.randint(0, 2**31 - 1),
        x_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state(),
        matmul_precision=torch.get_float32_matmul_precision(),
        num_threads=torch.get_num_threads(),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        use_duck_shape=torch.fx.experimental._config.use_duck_shape,
    )

    # Shared tensor between Learner and Actor, for the Learner to signal the Actor to stop
    x_stop = torch.tensor(False).share_memory_()

    # Start Actor process and its loop
    ctx = mp.get_context(_SPAWN)
    conn_read, conn_write = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_actor_loop, args=(weight_buffer, x_stop, conn_write, actor_config)
    )
    try:
        process.start()
    except BaseException:
        conn_read.close()
        conn_write.close()
        raise

    # Close the Learner's copy of the write end. Only the Actor can write to it
    conn_write.close()

    # Return Actor handle for the Learner
    return ActorHandle(process, conn_read, x_stop, weight_buffer)


def publish(
    actor_handle: ActorHandle, model: ActorCritic, version: int, ascension_current: float
) -> None:
    """Make the Learner's current weights version `version`.

    Publish once after startup, then once after each received rollout. The Actor can keep
    playing the previous version while the Learner updates the next one.

    Parameters
    ----------
    actor_handle : ActorHandle
        The handle returned by `start_actor`; its older weight slot is overwritten.
    model : ActorCritic
        The Learner's model.
    version : int
        The weights' version: the number of updates they have had.
    ascension_current : float
        The curriculum position the Actor should play them at.
    """
    if bool(actor_handle._x_stop):
        raise RuntimeError("The Actor has been stopped")
    
    if not actor_handle._process.is_alive():
        raise _died(actor_handle)
    
    write_weights(actor_handle._weight_buffer, model.state_dict(), version, ascension_current)


def receive_rollout(actor_handle: ActorHandle) -> tuple[Delivery, float]:
    """Return the next rollout and how long the Learner waited for it.

    A dead Actor surfaces here.

    Parameters
    ----------
    actor_handle : ActorHandle
        The Learner's handle on the Actor process.

    Returns
    -------
    delivery : Delivery
        The next rollout and how it was made.
    seconds_wait_learner : float
        How long the Learner waited for it, in seconds.
    """
    if bool(actor_handle._x_stop):
        raise RuntimeError("The Actor has been stopped")
    t_start = time.perf_counter()

    # Check if the Actor process is still alive
    while not actor_handle._conn_read.poll(timeout=1.0):
        if not actor_handle._process.is_alive():
            raise _died(actor_handle)

    try:
        delivery = actor_handle._conn_read.recv()

    except (EOFError, OSError):
        # It died before or while sending a rollout
        raise _died(actor_handle) from None

    return delivery, time.perf_counter() - t_start


def stop_actor(actor_handle: ActorHandle) -> None:
    """Ask the Actor to stop after its current rollout and wait for it.

    If it hangs, it is killed with its workers. It stops its own workers, which also exit on
    their own once it's gone. One waiting for a version sees the flag within a second.
    Calling it again has no effect.

    Parameters
    ----------
    actor_handle : ActorHandle
        The Learner's handle on the Actor process: its stop flag is set and its pipe end closed.
    """
    actor_handle._x_stop.fill_(True)
    actor_handle._conn_read.close()  # Ends a delivery it may be sending
    actor_handle._process.join(timeout=60.0)
    if actor_handle._process.is_alive():  # Likely stuck on a hung worker: kill its whole group
        os.killpg(actor_handle._process.pid, signal.SIGKILL)
        actor_handle._process.join()


def _actor_loop(
    weight_buffer: _WeightBuffer,
    x_stop: torch.Tensor,
    conn_write: Connection,
    actor_config: _ActorConfig,
) -> None:
    """Collect rollouts on the published weights, one version behind the Learner.

    It collects until stopped or orphaned. An exception ends the process, and the Learner's
    receive_rollout() reports it.

    Parameters
    ----------
    weight_buffer : _WeightBuffer
        The weights the Learner publishes, in shared memory.
    x_stop : torch.Tensor
        () bool. The stop flag, in shared memory.
    conn_write : Connection
        This process's end of the pipe the deliveries go through.
    actor_config : _ActorConfig
        The Actor process's config, fixed for the run.
    """
    # A group of its own, which its workers join
    os.setpgid(0, 0)

    # The processes this one starts mustn't hold the pipe open, or a dead Actor wouldn't end it
    os.set_inheritable(conn_write.fileno(), False)

    # Torch's default allocator before CUDA starts: the Learner can then open this process's
    # rollouts, which containers forbid for memory from the expandable-segments allocator
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    # Set the Actor's visible device
    device = actor_config.device
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # Set PyTorch config
    torch.set_float32_matmul_precision(actor_config.matmul_precision)
    torch.set_num_threads(actor_config.num_threads)
    torch.use_deterministic_algorithms(actor_config.deterministic)
    torch.fx.experimental._config.use_duck_shape = actor_config.use_duck_shape

    # Instance model
    model = ActorCritic(**actor_config.model_config).to(device)
    model.eval()

    # Its sampling continues the Learner's global stream, which the run seeds per rank and resume
    generator = torch.Generator(device)
    generator.set_state(actor_config.x_rng)

    # Get first weight versions
    if not _wait_for_version(weight_buffer.sph_new_version, x_stop):
        return

    weight_slot_first = max(weight_buffer.weight_slots, key=lambda slot: int(slot.x_version))
    version_first = int(weight_slot_first.x_version)

    with ExitStack() as stack:
        workers = start_workers(
            actor_config.rollout_config.num_envs,
            actor_config.rollout_config.num_workers,
            float(weight_slot_first.x_ascension_current),
            actor_config.gamma,
            actor_config.seed,
        )
        stack.callback(stop_workers, workers)

        # Loop
        version_rolled, t_idle = version_first, time.perf_counter()
        for rollout in itertools.count(version_first):
            version = max(rollout - 1, version_first)

            # A version not played yet is waited for: its publish's release orders the reads below
            if bool(x_stop) or (
                version != version_rolled
                and not _wait_for_version(weight_buffer.sph_new_version, x_stop)
            ):
                return

            # Get the version's weights and load them into the `ActorCritic` model
            weight_slot = next(
                slot for slot in weight_buffer.weight_slots if int(slot.x_version) == version
            )
            model.load_state_dict(weight_slot.weights)

            # Write the version's ascensions, which a game takes when it resets
            write_ascensions(workers, float(weight_slot.x_ascension_current))

            # Collect the rollout using the version's weights
            t_collect = time.perf_counter()
            rollout, episode_results = _collect_rollout(
                model,
                workers,
                actor_config.rollout_config,
                device,
                generator,
            )
            bytes_reserved_peak = 0
            if device.type == "cuda":
                torch.cuda.synchronize(device)  # The Learner reads the rollout in another process
                bytes_reserved_peak = torch.cuda.max_memory_reserved(device)
            
            delivery = Delivery(
                rollout,
                episode_results,
                seconds_collect=time.perf_counter() - t_collect,
                seconds_wait_actor=t_collect - t_idle,
                bytes_reserved_peak=bytes_reserved_peak,
            )
            t_idle = time.perf_counter()

            # Send the collected rollout over to the Actor
            try:
                conn_write.send(delivery)

            except BrokenPipeError:
                # The Learner stopped reading: it's stopping, or gone
                return

            # Update played version
            version_rolled = version


def _collect_rollout(
    model: ActorCritic,
    workers: Workers,
    rollout_config: RolloutConfig,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[Rollout, list[EpisodeResult]]:
    """Play a rollout's T lockstep steps of its E games with `model`, sampling from `generator`.

    Each step runs one forward over all games, then the workers step them. GAE is the Learner's.

    Parameters
    ----------
    model : ActorCritic
        The Actor's model.
    workers : Workers
        The handle to the workers that step the rollout's E games.
    rollout_config : RolloutConfig
        The rollout's games, E, and steps, T.
    device : torch.device
        The model's device.
    generator : torch.Generator
        The generator the actions are sampled from.

    Returns
    -------
    rollout : Rollout
        The T steps of the E games, before GAE.
    episode_results : list[EpisodeResult]
        The episodes finished during the rollout.
    """
    state_batches: list[StateBatch] = []  # Per step, in state order
    x_count_cpus: list[torch.Tensor] = []  # Their counts, on the CPU
    x_action_batches: list[torch.Tensor] = []  # Per step, (E, 3) on the device
    x_log_prob_batches: list[torch.Tensor] = []  # Per step, (E,) on the device
    x_reward_batches: list[torch.Tensor] = []  # Per step, (E, K) on the CPU
    x_done_batches: list[torch.Tensor] = []  # Per step, (E,) on the CPU
    episode_results: list[EpisodeResult] = []

    # Turn off gradient computation
    with torch.no_grad():
        state_batch, x_counts_cpu = snapshot_to_device(workers, device)
        for _ in range(rollout_config.rollout_length):
            # Forward pass, policy only
            policy_output = model.act(state_batch, generator=generator)
            x_step_actions = policy_output.x_actions

            # Step the worker's environments
            x_step_rewards, x_step_dones, step_episode_results = step_workers(
                workers, x_step_actions
            )

            # Append the step's data
            state_batches.append(state_batch)
            x_count_cpus.append(x_counts_cpu)
            x_action_batches.append(x_step_actions)
            x_log_prob_batches.append(policy_output.x_log_prob.sum(-1))
            x_reward_batches.append(x_step_rewards)
            x_done_batches.append(x_step_dones)
            episode_results.extend(step_episode_results)

            # Snapshot new state and Jagged counts for the next step
            state_batch, x_counts_cpu = snapshot_to_device(workers, device)

    return (
        Rollout(
            state_batch=cat_states(state_batches),
            x_counts_cpu=torch.cat(x_count_cpus),
            x_actions=torch.cat(x_action_batches),
            x_log_probs_behavior=torch.cat(x_log_prob_batches),
            x_rewards=torch.stack(x_reward_batches).to(device),
            x_dones=torch.stack(x_done_batches).to(device),
            state_batch_final=state_batch,
        ),
        episode_results,
    )


def _wait_for_version(sph_new_version: Semaphore, x_stop: torch.Tensor) -> bool:
    """Wait for the Learner's next publish.

    Parameters
    ----------
    sph_new_version : Semaphore
        The semaphore each publish releases once.
    x_stop : torch.Tensor
        () bool. The stop flag, in shared memory.

    Returns
    -------
    bool
        False once stopped, or if the Learner is gone; True otherwise.
    """
    while not sph_new_version.acquire(timeout=1.0):
        if bool(x_stop):
            # Stopped while it waits: stop_actor doesn't wake it
            return False

        if not parent_alive():
            # The Learner died without stopping this Actor
            return False

    return not x_stop


def _died(actor_handle: ActorHandle) -> RuntimeError:
    """Reap a failed Actor and describe its exit status.

    Parameters
    ----------
    actor_handle : ActorHandle
        The handle whose process died or closed its delivery pipe.

    Returns
    -------
    RuntimeError
        The failure to report to the Learner.
    """
    actor_handle._process.join(timeout=5.0)

    return RuntimeError(
        f"The Actor process died (exitcode {actor_handle._process.exitcode}); "
        "see its traceback above"
    )
