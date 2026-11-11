import random
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass

import torch
from torch.utils.tensorboard import SummaryWriter

from src.rl.algorithms.actor_critic import distributed
from src.rl.algorithms.actor_critic.actor import RolloutConfig
from src.rl.algorithms.actor_critic.actor import publish
from src.rl.algorithms.actor_critic.actor import receive_rollout
from src.rl.algorithms.actor_critic.actor import start_actor
from src.rl.algorithms.actor_critic.actor import stop_actor
from src.rl.algorithms.actor_critic.checkpoint import experiment_dir
from src.rl.algorithms.actor_critic.checkpoint import load_checkpoint
from src.rl.algorithms.actor_critic.checkpoint import save_checkpoint
from src.rl.algorithms.actor_critic.curriculum import AscensionCurriculum
from src.rl.algorithms.actor_critic.curriculum import step_curriculum
from src.rl.algorithms.actor_critic.distributed import RankInfo
from src.rl.algorithms.actor_critic.evals import Evaluator
from src.rl.algorithms.actor_critic.evals import poll_eval
from src.rl.algorithms.actor_critic.evals import start_eval
from src.rl.algorithms.actor_critic.evals import stop_evaluator
from src.rl.algorithms.actor_critic.evals import submit_eval
from src.rl.algorithms.actor_critic.metrics import MetricKey
from src.rl.algorithms.actor_critic.metrics import log_eval
from src.rl.algorithms.actor_critic.metrics import log_iteration
from src.rl.algorithms.actor_critic.ppo import PPOConfig
from src.rl.algorithms.actor_critic.ppo import adapt_lr_scale
from src.rl.algorithms.actor_critic.ppo import update_ppo
from src.rl.models import ActorCritic
from src.rl.models.actor_critic import compile_hot_paths
from src.rl.types import RunState
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

    # Announce train start
    if rank_info.rank == 0:
        _print_start(
            exp_name, model, iteration_start, run_state, rollout_config, rank_info.world_size
        )

    # Broadcast model's weights to the rest of the GPUs
    distributed.broadcast_from_rank0(model)

    # Calculate number of steps per iteration
    steps_per_iteration = (
        rank_info.world_size * rollout_config.rollout_length * rollout_config.num_envs
    )

    # Adaptive-KL learning rate: base recovered from the (possibly resumed-scaled) learning rate
    base_lr = optimizer.param_groups[0]["lr"] / run_state.lr_scale
    with ExitStack() as stack:
        writer = None
        evaluator = None

        # Rank 0 alone logs, checkpoints and evaluates: greedy fixed-seed games in a side process
        if rank_info.rank == 0:
            writer = stack.enter_context(SummaryWriter(experiment_dir(exp_name)))
            evaluator = start_eval(model_config, gamma)
            stack.callback(stop_evaluator, evaluator)

        # The Actor process plays one update behind; the Learner's own RNG keeps runs reproducible
        actor_handle = start_actor(
            model,
            model_config=model_config,
            rollout_config=rollout_config,
            device=rank_info.device,
            gamma=gamma,
        )

        # A failure must still stop the Actor: this process's exit would wait for it
        stack.callback(stop_actor, actor_handle)

        # The weights the Actor plays: the Learner publishes them, the Actor reads them
        publish(actor_handle, model, iteration_start, run_state.ascension_current)

        # Compile Learner's model only iff training on GPU: the Actor's model stays eager
        if rank_info.device.type == _DEVICE_TYPE_CUDA:
            compile_hot_paths(model)

        # Instance generator for batch sampling 
        generator = torch.Generator().manual_seed(random.randint(0, 2**31 - 1))

        # Let's go
        t_iteration = time.perf_counter()
        for iteration in range(iteration_start, num_iterations):
            # Receive Actor's rollout
            delivery, seconds_wait_learner = receive_rollout(actor_handle)

            # Update run state's total steps ran
            run_state.total_steps += steps_per_iteration

            # PPO update on the collected rollout
            t_update = time.perf_counter()
            metrics = update_ppo(
                model, delivery.rollout, optimizer, ppo_config, gamma, rank_info.device, generator
            )
            seconds_update = time.perf_counter() - t_update

            # Adaptive-KL-LR: the early stops bound epochs, not the first epoch's step size
            if ppo_config.target_kl is not None:
                run_state.lr_scale = adapt_lr_scale(
                    run_state.lr_scale, metrics[MetricKey.UPDATE_APPROX_KL], ppo_config.target_kl
                )
                for group in optimizer.param_groups:
                    group["lr"] = base_lr * run_state.lr_scale

            # Rank 0: log a finished eval battery and step the curriculum on its win rate
            if evaluator is not None and writer is not None:
                _finish_eval(evaluator, writer, run_state, curriculum)

            # Broadcast rank 0's curriculum position to the rest of the GPUs
            run_state.ascension_current = distributed.broadcast_float(
                run_state.ascension_current, rank_info.device
            )

            # Publish the updated weights to the Actor
            publish(actor_handle, model, iteration + 1, run_state.ascension_current)

            # Rank 0: submit an eval battery when one is due
            if evaluator is not None and iteration % cadence_config.eval_every == 0:
                submit_eval(evaluator, model, iteration, run_state.ascension_current)

            # Measure the iteration's samples per second
            t_now = time.perf_counter()
            samples_per_s = steps_per_iteration / (t_now - t_iteration)
            t_iteration = t_now

            # Log when due: every rank sends its finished episodes, rank 0 writes
            if iteration % cadence_config.log_every == 0:
                episode_results = distributed.gather_episodes(delivery.episodes)
                if writer is not None:
                    log_iteration(
                        writer,
                        iteration,
                        run_state,
                        metrics,
                        episode_results,
                        delivery,
                        seconds_update,
                        seconds_wait_learner,
                        samples_per_s,
                        rank_info.device,
                    )

            # A stop request on any rank stops every rank here, between collectives, after a save
            stopping = distributed.any_rank(stop.is_set(), rank_info.device)
            save_due = iteration % cadence_config.save_every == 0
            archive_due = iteration % cadence_config.archive_every == 0
            if rank_info.rank == 0 and (stopping or save_due or archive_due):
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
    n_params = sum(x_param.numel() for x_param in model.parameters())
    print(f"  params={n_params:,} ({n_params / 1e6:.2f}M)")
    print(
        f"  num_envs={rollout_config.num_envs}, "
        f"rollout_length={rollout_config.rollout_length}, "
        f"num_workers={rollout_config.num_workers}"
    )


def _finish_eval(
    evaluator: Evaluator,
    writer: SummaryWriter,
    run_state: RunState,
    curriculum: AscensionCurriculum,
) -> None:
    """Log an eval battery, if one has finished, and step the curriculum on its win rate.

    Parameters
    ----------
    evaluator : Evaluator
        The eval process.
    writer : SummaryWriter
        Rank 0's writer.
    run_state : RunState
        The run's progress, its curriculum position and streak stepped in place.
    curriculum : AscensionCurriculum
        The curriculum's settings.
    """
    # A finished eval battery, polled without blocking, logs at its own iteration
    battery = poll_eval(evaluator)
    if battery is None:
        return

    iteration_eval, ascension_current_eval, episode_results_eval = battery
    metrics_eval = log_eval(writer, iteration_eval, ascension_current_eval, episode_results_eval)
    step_curriculum(
        run_state, metrics_eval[MetricKey.WIN_RATE], ascension_current_eval, curriculum
    )
