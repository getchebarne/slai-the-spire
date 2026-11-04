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

import slai
import torch
import torch.multiprocessing as mp

from src.rl.algorithms.actor_critic.curriculum import ascensions
from src.rl.algorithms.actor_critic.stats import EpisodeStats
from src.rl.algorithms.actor_critic.workers import StepBuffer
from src.rl.algorithms.actor_critic.workers import WorkerHandle
from src.rl.algorithms.actor_critic.workers import drain_episodes
from src.rl.algorithms.actor_critic.workers import snapshot_to_device
from src.rl.algorithms.actor_critic.workers import start_workers
from src.rl.algorithms.actor_critic.workers import step_workers
from src.rl.algorithms.actor_critic.workers import stop_workers
from src.rl.models import ActorCritic
from src.rl.packed import cat_states
from src.rl.reward import RewardStream
from src.rl.types import X_GameState
from src.rl.types import X_Rollout
from src.rl.utils import parent_alive


_ACTION_TYPE_REST = int(slai.ActionType.Rest)
_ACTION_TYPE_SMITH = int(slai.ActionType.RestSmith)
_SPAWN = "spawn"

@dataclass
class RolloutConfig:
    num_envs: int
    rollout_workers: int
    rollout_length: int


@dataclass
class Delivery:
    """One collected rollout and how it was made."""

    x_rollout: X_Rollout
    episodes: list[EpisodeStats]
    metrics: dict[str, float]
    seconds_collect: float
    seconds_wait: float  # The Actor's idle time before it: waiting for weights or the Learner


@dataclass
class WeightSlot:
    """One slot the Learner publishes weights into, with their version and curriculum position."""

    x_version: torch.Tensor  # () int64, in shared memory

    # In shared CPU memory: containers block sharing the GPU memory of the Learner's allocator
    x_weights: dict[str, torch.Tensor]
    x_ascension_current: torch.Tensor  # () float64, in shared memory


@dataclass
class WeightBuffer:
    """The weights the Learner publishes and the Actor plays: two slots, a new-version signal."""

    # Two suffice: the Learner overwrites the older slot only after taking rollouts played on it
    weight_slots: list[WeightSlot]  # Last two weights published by the Learner
    sph_new_version: Semaphore  # Released once per publish; the Actor acquires once per version


@dataclass
class ActorHandle:
    """The Learner's handle on the Actor process: its process, pipe read end and stop flag."""

    process: BaseProcess
    conn_read: Connection  # FIFO; holds at most two Deliveries
    x_stop: torch.Tensor  # () bool in shared memory, lock-free like the workers' stop flag


@dataclass
class ActorConfig:
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


def make_weight_buffer(model: ActorCritic) -> WeightBuffer:
    """Return a weight buffer sized for `model`'s weights, with both slots unpublished.

    Parameters
    ----------
    model : ActorCritic
        The Learner's model.

    Returns
    -------
    WeightBuffer
        The buffer, in shared CPU memory.
    """
    return WeightBuffer(
        weight_slots=[
            WeightSlot(
                x_version=torch.tensor(-1).share_memory_(),
                x_weights={
                    n: torch.empty_like(x, device="cpu").share_memory_()
                    for n, x in model.state_dict().items()
                },
                x_ascension_current=torch.tensor(0.0, dtype=torch.float64).share_memory_(),
            )
            for _ in range(2)
        ],
        sph_new_version=mp.get_context(_SPAWN).Semaphore(0),
    )


def start_actor(
    weight_buffer: WeightBuffer,
    model_config: dict[str, Any],
    rollout_config: RolloutConfig,
    device: torch.device,
    gamma: float,
) -> ActorHandle:
    """Start the Actor process, playing the weights the Learner publishes into `weight_buffer`.

    The Actor process starts its workers and collects on a model of its own, from the
    Learner's random stream and torch settings.

    Parameters
    ----------
    weight_buffer : WeightBuffer
        The buffer the Learner publishes the weights into.
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
    # Instance the Actor's config. Read by the Actor loop 
    actor_config = ActorConfig(
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
    process.start()

    # Close the Learner's copy of the write end. Only the Actor can write to it
    conn_write.close()

    # Return Actor handle for the Learner
    return ActorHandle(process, conn_read, x_stop)


def publish(
    weight_buffer: WeightBuffer, model: ActorCritic, version: int, ascension_current: float
) -> None:
    """Make the Learner's current weights version `version`.

    Parameters
    ----------
    weight_buffer : WeightBuffer
        The buffer the Actor plays from; its older slot is overwritten.
    model : ActorCritic
        The Learner's model.
    version : int
        The weights' version: the number of updates they have had.
    ascension_current : float
        The curriculum position the Actor should play them at.
    """
    older = min(weight_buffer.weight_slots, key=lambda ws: int(ws.x_version))
    for name, x in model.state_dict().items():
        older.x_weights[name].copy_(x)  # A copy to the CPU's pageable memory returns once done
    older.x_ascension_current.fill_(ascension_current)
    older.x_version.fill_(version)
    weight_buffer.sph_new_version.release()


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
    seconds_learner_wait : float
        How long the Learner waited for it, in seconds.
    """
    t_start = time.perf_counter()

    # Check if the Actor process is still alive
    while not actor_handle.conn_read.poll(timeout=1.0):
        if not actor_handle.process.is_alive():
            raise _died(actor_handle)

    try:
        delivery = actor_handle.conn_read.recv()
    
    except (EOFError, OSError):  
        # It died before or while sending a rollout
        raise _died(actor_handle) from None

    return delivery, time.perf_counter() - t_start


def stop_actor(actor_handle: ActorHandle) -> None:
    """Ask the Actor to stop after its current rollout and wait for it.

    If it hangs, it is killed with its workers. It stops its own workers, which also exit on
    their own once it's gone. One waiting for a version sees the flag within a second.

    Parameters
    ----------
    actor_handle : ActorHandle
        The Learner's handle on the Actor process: its stop flag is set and its pipe end closed.
    """
    actor_handle.x_stop.fill_(True)
    actor_handle.conn_read.close()  # Ends a delivery it may be sending
    actor_handle.process.join(timeout=60.0)
    if actor_handle.process.is_alive():  # Likely stuck on a hung worker: kill its whole group
        os.killpg(actor_handle.process.pid, signal.SIGKILL)
        actor_handle.process.join()


def _died(actor_handle: ActorHandle) -> RuntimeError:
    actor_handle.process.join(timeout=5.0)

    return RuntimeError(
        f"The actor process died (exitcode {actor_handle.process.exitcode}); "
        "see its traceback above"
    )


def _actor_loop(
    weight_buffer: WeightBuffer,
    x_stop: torch.Tensor,
    conn_write: Connection,
    actor_config: ActorConfig,
) -> None:
    """Collect rollouts on the published weights, one version behind the Learner.

    It collects until stopped or orphaned. An exception ends the process, and the Learner's
    receive_rollout() reports it.

    Parameters
    ----------
    weight_buffer : WeightBuffer
        The weights the Learner publishes, in shared memory.
    x_stop : torch.Tensor
        () bool. The stop flag, in shared memory.
    conn_write : Connection
        This process's end of the pipe the deliveries go through.
    actor_config : ActorConfig
        The Actor process's config, fixed for the run.
    """
    # A group of its own, which its workers join
    os.setpgid(0, 0)

    # The processes this one starts mustn't hold the pipe open, or a dead Actor wouldn't end it
    os.set_inheritable(conn_write.fileno(), False)

    # Torch's default allocator before CUDA starts: the Learner can then open this process's
    # rollouts, which containers forbid for memory from the expandable-segments allocator
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    device = actor_config.device
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision(actor_config.matmul_precision)
    torch.set_num_threads(actor_config.num_threads)
    torch.use_deterministic_algorithms(actor_config.deterministic)
    torch.fx.experimental._config.use_duck_shape = actor_config.use_duck_shape

    # Instance model
    model = ActorCritic(**actor_config.model_config).to(device)

    # Its sampling continues the Learner's global stream, which the run seeds per rank and resume
    generator = torch.Generator(device)
    generator.set_state(actor_config.x_rng)

    # Get first weight versions
    if not _wait_for_version(weight_buffer.sph_new_version, x_stop):
        return
    
    weight_slot_first = max(weight_buffer.weight_slots, key=lambda slot: int(slot.x_version))
    version_first = int(weight_slot_first.x_version)

    with ExitStack() as stack:
        
        # Start Workers
        step_buffer, worker_handles = start_workers(
            actor_config.rollout_config.num_envs,
            actor_config.rollout_config.rollout_workers,
            float(weight_slot_first.x_ascension_current),
            actor_config.gamma,
            actor_config.seed,
        )
        stack.callback(stop_workers, worker_handles)

        # Loop
        version_rolled, t_idle = version_first, time.perf_counter()  # Already waited for
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
            model.load_state_dict(weight_slot.x_weights)

            # Collect the rollout using the version's weights
            t_collect = time.perf_counter()
            x_rollout, episodes, metrics = _collect_rollout(
                model,
                step_buffer,
                worker_handles,
                actor_config.rollout_config,
                device,
                float(weight_slot.x_ascension_current),
                generator,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)  # The Learner reads the rollout in another process
                metrics["memory/actor_gb"] = torch.cuda.max_memory_reserved(device) / 2**30
            delivery = Delivery(
                x_rollout,
                episodes,
                metrics,
                seconds_collect=time.perf_counter() - t_collect,
                seconds_wait=t_collect - t_idle,
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
    step_buffer: StepBuffer,
    worker_handles: list[WorkerHandle],
    rollout_config: RolloutConfig,
    device: torch.device,
    ascension_current: float,
    generator: torch.Generator,
) -> tuple[X_Rollout, list[EpisodeStats], dict[str, float]]:
    """Play a rollout's T lockstep steps of its E games with `model`, sampling from `generator`.

    Each step runs one forward over all games, then the workers step them. GAE is the Learner's.

    Parameters
    ----------
    model : ActorCritic
        The Actor's model, in eval mode during the rollout and back in training mode after it.
    step_buffer : StepBuffer
        What the Actor and its workers exchange; its step holds the E games.
    worker_handles : list[WorkerHandle]
        The worker processes that step the games.
    rollout_config : RolloutConfig
        The rollout's games, E, and steps, T.
    device : torch.device
        The model's device.
    ascension_current : float
        The soft-curriculum position, which sets the ascension of a game that resets.
    generator : torch.Generator
        The generator the actions are sampled from.

    Returns
    -------
    x_rollout : X_Rollout
        The T steps of the E games, before GAE.
    episodes : list[EpisodeStats]
        The finished episodes, drained from the workers' queue.
    metrics : dict[str, float]
        The rollout's scalars: its terminations (step-limit cuts included), boundary truncations and
        Rest Site upgrade rate.
    """
    step_buffer.x_ascensions.copy_(
        torch.tensor(ascensions(ascension_current, rollout_config.num_envs))
    )
    x_game_states: list[X_GameState] = []  # Per step, in state order
    x_count_cpus: list[torch.Tensor] = []  # Their counts, on the CPU
    x_actions = torch.empty(
        rollout_config.rollout_length * rollout_config.num_envs, 4, dtype=torch.long, device=device
    )
    x_log_probs_behavior = torch.empty(
        rollout_config.rollout_length * rollout_config.num_envs, device=device
    )
    x_rewards = torch.empty(
        rollout_config.rollout_length, rollout_config.num_envs, len(RewardStream), device=device
    )
    x_dones = torch.empty(rollout_config.rollout_length, rollout_config.num_envs, device=device)

    # Rest-site preference: device-side counters, read once after the loop (no per-step sync)
    x_rest_selections = torch.zeros((), device=device)
    x_upgrade_selections = torch.zeros((), device=device)

    # Set model to eval, turn off gradient computation
    model.eval()
    with torch.no_grad():
        x_game_state, x_count_cpu = snapshot_to_device(step_buffer, device)
        for t in range(rollout_config.rollout_length):
            x_action = model.act(x_game_state, generator=generator)
            x_game_states.append(x_game_state)
            x_count_cpus.append(x_count_cpu)
            step_rows = slice(t * rollout_config.num_envs, (t + 1) * rollout_config.num_envs)
            x_actions[step_rows] = torch.stack(
                [
                    x_action.x_action_type,
                    x_action.x_token_kind,
                    x_action.x_rank,
                    x_action.x_target,
                ],
                dim=1,
            )
            x_log_probs_behavior[step_rows] = x_action.x_log_prob.sum(-1)

            # Campfire decision: both Rest and RestSmith legal — which was chosen?
            x_both = (
                x_game_state.x_legal_action_type[:, _ACTION_TYPE_REST]
                & x_game_state.x_legal_action_type[:, _ACTION_TYPE_SMITH]
            )
            x_rest_selections.add_((x_both & (x_action.x_action_type == _ACTION_TYPE_REST)).sum())
            x_upgrade_selections.add_(
                (x_both & (x_action.x_action_type == _ACTION_TYPE_SMITH)).sum()
            )

            step_buffer.x_actions.copy_(x_actions[step_rows])
            step_workers(worker_handles)
            x_rewards[t] = step_buffer.x_rewards
            x_dones[t] = step_buffer.x_dones
            x_game_state, x_count_cpu = snapshot_to_device(step_buffer, device)

    # Set model to training mode
    model.train()

    # Surface the termination vs rollout-boundary-bootstrap split (the GAE truncation
    # path): how many envs were still mid-episode at T and relied on V(s_T).
    n_campfire_selections = (x_rest_selections + x_upgrade_selections).item()
    metrics = {
        "rollout/terminations": x_dones.sum().item(),
        "rollout/boundary_truncations": float(rollout_config.num_envs) - x_dones[-1].sum().item(),
        # 0.0 also when no campfire was seen this rollout
        "rollout/rest_upgrade_rate": x_upgrade_selections.item() / n_campfire_selections
        if n_campfire_selections
        else 0.0,
    }

    return (
        X_Rollout(
            x_game_state=cat_states(x_game_states),
            x_count_cpu=torch.cat(x_count_cpus),
            x_actions=x_actions,
            x_log_probs_behavior=x_log_probs_behavior,
            x_rewards=x_rewards,
            x_dones=x_dones,
            x_game_state_final=x_game_state,
        ),
        drain_episodes(step_buffer.episodes),
        metrics,
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
