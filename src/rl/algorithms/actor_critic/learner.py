"""The Learner: PPO on the Actor's rollouts, with the evals, the curriculum and the checkpoints."""

import random
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass

import torch
from torch.utils.tensorboard import SummaryWriter

from src.rl.algorithms.actor_critic import distributed
from src.rl.algorithms.actor_critic.actor import RolloutConfig
from src.rl.algorithms.actor_critic.actor import make_weight_buffer
from src.rl.algorithms.actor_critic.actor import publish
from src.rl.algorithms.actor_critic.actor import receive_rollout
from src.rl.algorithms.actor_critic.actor import start_actor
from src.rl.algorithms.actor_critic.actor import stop_actor
from src.rl.algorithms.actor_critic.checkpoint import RunState
from src.rl.algorithms.actor_critic.checkpoint import experiment_dir
from src.rl.algorithms.actor_critic.checkpoint import load_checkpoint
from src.rl.algorithms.actor_critic.checkpoint import save_checkpoint
from src.rl.algorithms.actor_critic.curriculum import AscensionCurriculum
from src.rl.algorithms.actor_critic.curriculum import step_curriculum
from src.rl.algorithms.actor_critic.distributed import RankInfo
from src.rl.algorithms.actor_critic.evals import EVAL_ACT2_RATE
from src.rl.algorithms.actor_critic.evals import EVAL_ACT3_RATE
from src.rl.algorithms.actor_critic.evals import EVAL_ACT4_RATE
from src.rl.algorithms.actor_critic.evals import EVAL_FLOOR_AVG
from src.rl.algorithms.actor_critic.evals import EVAL_RETURN_AVG
from src.rl.algorithms.actor_critic.evals import EVAL_WIN_RATE
from src.rl.algorithms.actor_critic.evals import Evaluator
from src.rl.algorithms.actor_critic.evals import poll_eval
from src.rl.algorithms.actor_critic.evals import start_eval
from src.rl.algorithms.actor_critic.evals import stop_evaluator
from src.rl.algorithms.actor_critic.evals import submit_eval
from src.rl.algorithms.actor_critic.ppo import PPOConfig
from src.rl.algorithms.actor_critic.ppo import adapt_lr_scale
from src.rl.algorithms.actor_critic.ppo import update_ppo
from src.rl.algorithms.actor_critic.stats import episode_scalars
from src.rl.models import ActorCritic
from src.rl.models.actor_critic import compile_hot_paths
from src.rl.utils import init_optimizer
from src.rl.utils import set_rng_seeds


_DEVICE_TYPE_CUDA = "cuda"


@dataclass
class CadenceConfig:
    """The run's cadences, in iterations.

    Every how many iterations the run logs, checkpoints, starts an eval and keeps a checkpoint
    as an archive.
    """

    log_every: int
    save_every: int
    eval_every: int
    archive_every: int


def train(
    exp_name: str,
    num_iterations: int,
    seed: int,
    cadence_config: CadenceConfig,
    model_config: dict,
    optimizer_config: dict,
    gamma: float,
    curriculum: AscensionCurriculum,
    ppo_config: PPOConfig,
    rollout_config: RolloutConfig,
    rank_info: RankInfo,
    stop: threading.Event,
) -> None:
    # Set seeds
    set_rng_seeds(seed)

    # Instance `ActorCritic` model and optimizer
    model = ActorCritic(**model_config).to(rank_info.device)
    optimizer = init_optimizer(optimizer_config["name"], model, **optimizer_config["kwargs"])

    # Load checkpoint
    iteration_start, run_state = load_checkpoint(exp_name, model, optimizer)
    if run_state is None:
        run_state = RunState(
            total_steps=0, ascension_current=curriculum.ascension_start, streak=0, lr_scale=1.0
        )

    # Offset the seeds per resume and rank; rank 0 of a fresh run keeps the single-GPU seeds
    if iteration_start > 0 or rank_info.rank > 0:
        set_rng_seeds(seed + iteration_start + 7919 * rank_info.rank)
    if rank_info.rank == 0:
        _print_start(
            exp_name, model, iteration_start, run_state, rollout_config, rank_info.world_size
        )
    distributed.broadcast_from_rank0(model)
    steps_per_iteration = (
        rank_info.world_size * rollout_config.rollout_length * rollout_config.num_envs
    )

    # Adaptive-KL lr: base recovered from the (possibly resumed-scaled) group lr
    base_lr = optimizer.param_groups[0]["lr"] / run_state.lr_scale
    with ExitStack() as stack:
        writer: SummaryWriter | None = None
        evaluator: Evaluator | None = None

        # Rank 0 alone logs, checkpoints and evaluates: greedy fixed-seed games in a side process
        if rank_info.rank == 0:
            writer = stack.enter_context(SummaryWriter(experiment_dir(exp_name)))
            evaluator = start_eval(model_config, gamma)
            stack.callback(stop_evaluator, evaluator)

        # The weights the Actor plays: the Learner publishes them, the Actor reads them
        weight_buffer = make_weight_buffer(model)
        publish(weight_buffer, model, iteration_start, run_state.ascension_current)

        # The Actor process plays one update behind; the Learner's own RNG keeps runs reproducible
        actor_handle = start_actor(
            weight_buffer,
            model_config=model_config,
            rollout_config=rollout_config,
            device=rank_info.device,
            gamma=gamma,
        )

        # A failure must still stop the Actor: this process's exit would wait for it
        stack.callback(stop_actor, actor_handle)

        # Learner only, GPU only: the Actor's model stays eager
        if rank_info.device.type == _DEVICE_TYPE_CUDA:
            compile_hot_paths(model)

        generator = torch.Generator().manual_seed(random.randint(0, 2**31 - 1))
        t_iteration = time.perf_counter()
        for iteration in range(iteration_start, num_iterations):
            delivery, seconds_learner_wait = receive_rollout(actor_handle)
            run_state.total_steps += steps_per_iteration

            # Update
            t_update = time.perf_counter()
            metrics = update_ppo(
                model, delivery.rollout, optimizer, ppo_config, gamma, rank_info.device, generator
            )
            seconds_update = time.perf_counter() - t_update

            # Adaptive-KL-LR: the early stops bound epochs, not the first epoch's step size
            if ppo_config.target_kl is not None:
                run_state.lr_scale = adapt_lr_scale(
                    run_state.lr_scale, metrics["update/approx_kl"], ppo_config.target_kl
                )
                for group in optimizer.param_groups:
                    group["lr"] = base_lr * run_state.lr_scale
                metrics["update/lr_scale"] = run_state.lr_scale

            # A finished eval battery, polled without blocking, logs at its own iteration
            battery = poll_eval(evaluator) if evaluator is not None else None
            if battery is not None:
                assert writer is not None
                iteration_eval, ascension_current_eval, metrics_eval = battery
                _log_eval(writer, iteration_eval, ascension_current_eval, metrics_eval)
                step_curriculum(
                    run_state, metrics_eval[EVAL_WIN_RATE], ascension_current_eval, curriculum
                )
            run_state.ascension_current = distributed.broadcast_float(
                run_state.ascension_current, rank_info.device
            )
            publish(weight_buffer, model, iteration + 1, run_state.ascension_current)
            t_now = time.perf_counter()
            samples_per_s = steps_per_iteration / (t_now - t_iteration)
            t_iteration = t_now

            if evaluator is not None and iteration % cadence_config.eval_every == 0:
                submit_eval(evaluator, model, iteration, run_state.ascension_current)

            if iteration % cadence_config.log_every == 0:
                episodes = distributed.gather_episodes(delivery.episodes)
                if writer is not None:
                    scalars = {
                        **metrics,
                        "train/ascension": run_state.ascension_current,
                        "Steps/total": run_state.total_steps,
                        "Time/rollout": delivery.seconds_collect,
                        "Time/update": seconds_update,
                        "pipeline/actor_wait": delivery.seconds_wait,
                        "pipeline/learner_wait": seconds_learner_wait,
                        "pipeline/samples_per_s": samples_per_s,
                        **delivery.metrics,
                        **episode_scalars(episodes),
                    }
                    _log_iteration(writer, iteration, scalars, len(episodes), rank_info.device)

            # A stop request on any rank stops every rank here, between collectives, after a save
            stopping = distributed.any_rank(stop.is_set(), rank_info.device)
            if rank_info.rank == 0 and (
                stopping
                or iteration % cadence_config.save_every == 0
                or iteration % cadence_config.archive_every == 0
            ):
                if stopping:
                    print("\nTraining interrupted. Saving checkpoint...")

                save_checkpoint(
                    exp_name, model, optimizer, iteration, run_state, cadence_config.archive_every
                )
            if stopping:
                break


def _print_start(
    exp_name: str,
    model: ActorCritic,
    iteration_start: int,
    run_state: RunState,
    rollout_config: RolloutConfig,
    world_size: int,
) -> None:
    """Print the run's start: where it resumes from, its model's size and its rollouts' shape.

    Parameters
    ----------
    exp_name : str
        The run's name.
    model : ActorCritic
        The Learner's model.
    iteration_start : int
        The first iteration to train, 0 for a fresh run.
    run_state : RunState
        The run's progress.
    rollout_config : RolloutConfig
        The rollouts' shape.
    world_size : int
        The number of ranks.
    """
    if iteration_start > 0:
        print(
            f"Resuming from {experiment_dir(exp_name)}/checkpoints/"
            f"iter-{iteration_start - 1}.pth: iteration {iteration_start}, "
            f"{run_state.total_steps} steps"
        )
    print(f"Starting training: {exp_name} on {world_size} rank(s)")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:,} ({n_params / 1e6:.2f}M)")
    print(
        f"  num_envs={rollout_config.num_envs}, "
        f"rollout_length={rollout_config.rollout_length}, "
        f"rollout_workers={rollout_config.rollout_workers}"
    )


def _log_eval(
    writer: SummaryWriter,
    iteration_eval: int,
    ascension_current_eval: float,
    metrics_eval: dict[str, float],
) -> None:
    """Write a finished eval battery's scalars at its iteration and print its line.

    Parameters
    ----------
    writer : SummaryWriter
        Rank 0's writer.
    iteration_eval : int
        The iteration whose weights the battery played.
    ascension_current_eval : float
        The curriculum position it played at.
    metrics_eval : dict[str, float]
        Its eval/* metrics.
    """
    for key, value in metrics_eval.items():
        writer.add_scalar(key, value, iteration_eval)
    writer.add_scalar("eval/ascension", ascension_current_eval, iteration_eval)
    print(
        f"  eval@{iteration_eval} c={ascension_current_eval:.2f}: "
        f"win_rate={metrics_eval[EVAL_WIN_RATE]:.3f} "
        f"floor={metrics_eval[EVAL_FLOOR_AVG]:.2f} "
        f"act2={metrics_eval[EVAL_ACT2_RATE]:.2f} "
        f"act3={metrics_eval[EVAL_ACT3_RATE]:.2f} "
        f"act4={metrics_eval[EVAL_ACT4_RATE]:.2f} "
        f"return={metrics_eval[EVAL_RETURN_AVG]:.4f}"
    )


def _log_iteration(
    writer: SummaryWriter,
    iteration: int,
    scalars: dict[str, float],
    num_episodes: int,
    device: torch.device,
) -> None:
    """Print an iteration's line and write its scalars, with the Learner's memory peak on a GPU.

    Parameters
    ----------
    writer : SummaryWriter
        Rank 0's writer.
    iteration : int
        The iteration.
    scalars : dict[str, float]
        Its scalars, by name.
    num_episodes : int
        The episodes its rollouts finished, every rank's counted.
    device : torch.device
        The Learner's device.
    """
    print(
        f"Iter {iteration} | steps={scalars['Steps/total']} | "
        f"policy={scalars['loss/policy']:.4f} value={scalars['loss/value']:.4f} | "
        f"episodes={num_episodes}"
    )
    if device.type == _DEVICE_TYPE_CUDA:
        scalars["memory/learner_gb"] = torch.cuda.max_memory_reserved(device) / 2**30
    for key, value in scalars.items():
        writer.add_scalar(key, value, iteration)
