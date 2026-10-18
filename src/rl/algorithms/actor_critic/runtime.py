"""The actor process and its handoff: version v, the weights after v updates, plays rollout v+1"""

import itertools
import os
import signal
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Semaphore
from typing import Any

import torch
import torch.multiprocessing as mp

from src.rl.algorithms.actor_critic.curriculum import AscensionCurriculum
from src.rl.algorithms.actor_critic.episode import EpisodeStats
from src.rl.algorithms.actor_critic.rollout import collect_rollout
from src.rl.algorithms.actor_critic.workers import RolloutConfig
from src.rl.algorithms.actor_critic.workers import parent_alive
from src.rl.algorithms.actor_critic.workers import start_workers
from src.rl.algorithms.actor_critic.workers import stop_workers
from src.rl.models import ActorCritic
from src.rl.types import X_Rollout


@dataclass
class Delivery:
    """One collected rollout and how it was made"""

    rollout: X_Rollout
    completed: list[EpisodeStats]
    info: dict[str, float]
    seconds_collect: float
    seconds_wait: float  # the actor's idle time before it: waiting for weights or the learner


@dataclass
class Published:
    """One published weight set and the curriculum position to play it at"""

    version: torch.Tensor  # () int64, in shared memory

    # In shared CPU memory: containers block sharing the GPU memory of the learner's allocator
    weights: dict[str, torch.Tensor]
    curriculum_c: torch.Tensor  # () float64, in shared memory


@dataclass
class Handoff:
    """What the learner and its actor process share"""

    # Two suffice: the learner overwrites the older set only after taking the rollouts played on it
    published: list[Published]  # Last two weights published by the Learner
    new_version: Semaphore  # released once by each publish; the actor acquires once per version
    stop: torch.Tensor  # () bool in shared memory, lock-free like the workers' stop flag


@dataclass
class Actor:
    """The actor process and the learner's end of the pipe its deliveries come through"""

    process: BaseProcess
    conn: Connection  # FIFO; holds at most two Deliveries


@dataclass
class ActorSettings:
    """The settings of the actor process, fixed for the run"""

    model_config: dict[str, Any]
    rollout_config: RolloutConfig
    device: torch.device
    first_version: int
    curriculum: AscensionCurriculum
    gamma: float
    seed: int

    # The learner's random-generator state and torch settings, which the actor process starts from
    rng: torch.Tensor  # the global generator's state on the device
    matmul_precision: str
    num_threads: int
    deterministic: bool


def start_actor(model: ActorCritic, settings: ActorSettings) -> tuple[Handoff, Actor]:
    """Publish `model` as `settings.first_version` and start the actor process, which starts its
    workers and collects on a model of its own"""
    ctx = mp.get_context("spawn")  # CUDA is live in this process; fork is unsafe
    handoff = Handoff(
        published=[
            Published(
                version=torch.tensor(-1).share_memory_(),
                weights={
                    n: torch.empty_like(x, device="cpu").share_memory_()
                    for n, x in model.state_dict().items()
                },
                curriculum_c=torch.tensor(0.0, dtype=torch.float64).share_memory_(),
            )
            for _ in range(2)
        ],
        new_version=ctx.Semaphore(0),
        stop=torch.tensor(False).share_memory_(),
    )
    publish(handoff, model, settings.first_version, settings.curriculum.ascension_start)
    conn, child = ctx.Pipe(duplex=False)

    # Not a daemon: a daemon process can't start the workers
    process = ctx.Process(target=_actor_loop, args=(handoff, child, settings))
    process.start()
    child.close()  # the actor holds its own copy: a dead actor then ends the pipe
    return handoff, Actor(process, conn)


def publish(handoff: Handoff, model: ActorCritic, version: int, curriculum_c: float) -> None:
    """Make the learner's current weights version `version`, with the curriculum position
    the actor should play them at"""
    older = min(handoff.published, key=lambda p: int(p.version))
    for name, x in model.state_dict().items():
        older.weights[name].copy_(x)  # a copy to the CPU's pageable memory returns once done
    older.curriculum_c.fill_(curriculum_c)
    older.version.fill_(version)
    handoff.new_version.release()


def take(actor: Actor) -> tuple[Delivery, float]:
    """The next rollout and how long the learner waited for it. A dead actor surfaces here"""
    t_start = time.perf_counter()
    while not actor.conn.poll(timeout=1.0):
        if not actor.process.is_alive():
            raise _died(actor)
    try:
        delivery = actor.conn.recv()
    except (EOFError, OSError):  # it died before or while sending a rollout
        raise _died(actor) from None
    return delivery, time.perf_counter() - t_start


def stop_actor(handoff: Handoff, actor: Actor) -> None:
    """Ask the actor to stop after its current rollout and wait for it, killing it with its
    workers if it hangs. It stops its own workers, which also exit on their own once it's gone"""
    handoff.stop.fill_(True)
    handoff.new_version.release()  # wakes it if it waits for a version
    actor.conn.close()  # ends a delivery it may be sending
    actor.process.join(timeout=60.0)
    if actor.process.is_alive():  # likely stuck on a hung worker: kill its whole group
        os.killpg(actor.process.pid, signal.SIGKILL)
        actor.process.join()


def _died(actor: Actor) -> RuntimeError:
    actor.process.join(timeout=5.0)
    return RuntimeError(
        f"the actor process died (exitcode {actor.process.exitcode}); see its traceback above"
    )


def _actor_loop(handoff: Handoff, conn: Connection, settings: ActorSettings) -> None:
    """Collect rollouts on the published weights, one version behind the learner, until
    stopped or orphaned. An exception ends the process, and the learner's take() reports it"""

    # A group of its own, which its workers join: the run's Ctrl-C or SIGTERM reaches only the
    # learner, which stops this process after its checkpoint, and stop_actor can kill them all
    os.setpgid(0, 0)

    # The processes this one starts mustn't hold the pipe open, or a dead actor wouldn't end it
    os.set_inheritable(conn.fileno(), False)

    # Torch's default allocator before CUDA starts: the learner can then open this process's
    # rollouts, which containers forbid for memory from the expandable-segments allocator
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    device = settings.device
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision(settings.matmul_precision)
    torch.set_num_threads(settings.num_threads)
    torch.use_deterministic_algorithms(settings.deterministic)
    model = ActorCritic(**settings.model_config).to(device)  # its weights come from the handoff

    # Its sampling continues the learner's global stream, which the run seeds per rank and resume
    generator = torch.Generator(device)
    generator.set_state(settings.rng)
    exchange, workers = start_workers(
        settings.rollout_config, settings.curriculum, settings.gamma, settings.seed
    )
    try:
        played, t_idle = None, time.perf_counter()
        for i in itertools.count(settings.first_version):
            version = max(i - 1, settings.first_version)

            # A version not played yet is waited for: its publish's release orders the reads below
            if bool(handoff.stop) or (
                version != played and not _wait_for_version(handoff.new_version, handoff.stop)
            ):
                return
            played = version
            published = next(p for p in handoff.published if int(p.version) == version)
            model.load_state_dict(published.weights)
            t_collect = time.perf_counter()
            x_rollout, completed, info = collect_rollout(
                model,
                exchange,
                workers,
                settings.rollout_config.rollout_length,
                device,
                float(published.curriculum_c),
                generator,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)  # the learner reads the rollout in another process
                info["memory/actor_gb"] = torch.cuda.max_memory_reserved(device) / 2**30
            delivery = Delivery(
                x_rollout,
                completed,
                info,
                seconds_collect=time.perf_counter() - t_collect,
                seconds_wait=t_collect - t_idle,
            )
            t_idle = time.perf_counter()
            try:
                conn.send(delivery)  # past the pipe's buffer, waits for the learner to read it
            except BrokenPipeError:  # the learner stopped reading: it's stopping, or gone
                return
    finally:
        stop_workers(exchange, workers)


def _wait_for_version(new_version: Semaphore, stop: torch.Tensor) -> bool:
    """Wait for the learner's next publish: False once stopped, or if the learner is gone"""
    while not new_version.acquire(timeout=1.0):
        if not parent_alive():  # the learner died without stopping this actor
            return False
    return not stop
