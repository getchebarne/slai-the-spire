import argparse
import random
import signal
import threading
import time
from dataclasses import dataclass, replace

import torch
from torch.utils.tensorboard import SummaryWriter

from src.rl.algorithms.actor_critic import distributed
from src.rl.algorithms.actor_critic.checkpoint import archive_path
from src.rl.algorithms.actor_critic.checkpoint import checkpoint_path
from src.rl.algorithms.actor_critic.checkpoint import experiment_dir
from src.rl.algorithms.actor_critic.checkpoint import load_checkpoint
from src.rl.algorithms.actor_critic.checkpoint import make_experiment_dir
from src.rl.algorithms.actor_critic.checkpoint import save_checkpoint
from src.rl.algorithms.actor_critic.curriculum import AscensionCurriculum
from src.rl.algorithms.actor_critic.curriculum import frontier_level
from src.rl.algorithms.actor_critic.curriculum import step_curriculum
from src.rl.algorithms.actor_critic.distributed import RankInfo
from src.rl.algorithms.actor_critic.episode import episode_scalars
from src.rl.algorithms.actor_critic.evals import EVAL_ACT2_RATE
from src.rl.algorithms.actor_critic.evals import EVAL_FLOOR_AVG
from src.rl.algorithms.actor_critic.evals import EVAL_REWARD_AVG
from src.rl.algorithms.actor_critic.evals import EVAL_WIN_RATE
from src.rl.algorithms.actor_critic.evals import poll_eval
from src.rl.algorithms.actor_critic.evals import start_eval
from src.rl.algorithms.actor_critic.evals import stop_eval
from src.rl.algorithms.actor_critic.evals import submit_eval
from src.rl.algorithms.actor_critic.ppo import PPOConfig
from src.rl.algorithms.actor_critic.ppo import adapt_lr_scale
from src.rl.algorithms.actor_critic.ppo import prepare_batch
from src.rl.algorithms.actor_critic.ppo import update_ppo
from src.rl.algorithms.actor_critic.runtime import ActorSettings
from src.rl.algorithms.actor_critic.runtime import publish
from src.rl.algorithms.actor_critic.runtime import start_actor
from src.rl.algorithms.actor_critic.runtime import stop_actor
from src.rl.algorithms.actor_critic.runtime import take
from src.rl.algorithms.actor_critic.workers import RolloutConfig
from src.rl.models import ActorCritic
from src.rl.models.actor_critic import compile_hot_paths
from src.rl.utils import init_optimizer
from src.rl.utils import load_config
from src.rl.utils import set_rng_seeds


@dataclass
class CadenceConfig:
    """Every how many iterations the run logs, checkpoints, starts an eval and archives a copy
    of its checkpoint"""

    log_every: int
    save_every: int
    eval_every: int
    archive_every: int


def train(
    exp_name: str,
    num_iterations: int,
    cadence: CadenceConfig,
    model: ActorCritic,
    model_config: dict,
    optimizer: torch.optim.Optimizer,
    gamma: float,
    curriculum: AscensionCurriculum,
    ppo: PPOConfig,
    rollout_config: RolloutConfig,
    rank_info: RankInfo,
    stop: threading.Event,
    start_iteration: int,
    total_steps: int,
    lr_scale: float,
) -> None:
    # Rank 0 alone evaluates, logs and checkpoints
    writer = SummaryWriter(experiment_dir(exp_name)) if rank_info.rank == 0 else None
    distributed.broadcast_from_rank0(model)
    ckpt_path = checkpoint_path(exp_name)
    c = curriculum.ascension_start  # soft-curriculum position (see step_curriculum)
    streak = 0

    # Eval runs a fixed-seed greedy battery in a persistent side process
    evaluator = (
        start_eval(model_config, gamma, curriculum.ascension_max) if rank_info.rank == 0 else None
    )

    # Adaptive-KL lr: base recovered from the (possibly resumed-scaled) group lr
    base_lr = optimizer.param_groups[0]["lr"] / lr_scale

    # From here on a failure must still stop the actor: this process's exit would wait for it
    handoff = None
    try:
        # The actor process plays one update behind; the learner's own RNG keeps runs reproducible
        handoff, actor = start_actor(
            model,
            ActorSettings(
                model_config=model_config,
                rollout_config=rollout_config,
                device=rank_info.device,
                first_version=start_iteration,
                curriculum=curriculum,
                gamma=gamma,
                seed=random.randint(0, 2**31 - 1),
                rng=(
                    torch.cuda.get_rng_state(rank_info.device)
                    if rank_info.device.type == "cuda"
                    else torch.get_rng_state()
                ),
                matmul_precision=torch.get_float32_matmul_precision(),
                num_threads=torch.get_num_threads(),
                deterministic=torch.are_deterministic_algorithms_enabled(),
            ),
        )

        # The learner's model only, on the GPU: the actor process builds an eager model of its
        # own, and the CPU parity runs stay bit-identical
        if rank_info.device.type == "cuda":
            compile_hot_paths(model)

        # The ranks start averaging each bucket of gradients while the backward still runs
        buckets = distributed.hook_gradient_buckets(model)
        generator = torch.Generator(device=rank_info.device).manual_seed(
            random.randint(0, 2**31 - 1)
        )
        steps_per_iteration = (
            rank_info.world_size * rollout_config.rollout_length * rollout_config.num_envs
        )
        t_iteration = time.perf_counter()
        for iteration in range(start_iteration, num_iterations):
            delivery, seconds_learner_wait = take(actor)
            t_update = time.perf_counter()
            total_steps += steps_per_iteration
            batch = prepare_batch(model, delivery.rollout, gamma, ppo)
            metrics = update_ppo(
                model, batch, optimizer, buckets, ppo, rank_info.device, generator
            )
            seconds_update = time.perf_counter() - t_update

            # Adaptive-KL lr: the early stops bound epochs, not the first epoch's step size
            if ppo.target_kl is not None:
                lr_scale = adapt_lr_scale(lr_scale, metrics["update/approx_kl"], ppo.target_kl)
                for group in optimizer.param_groups:
                    group["lr"] = base_lr * lr_scale
                metrics["update/lr_scale"] = lr_scale

            # Eval battery (persistent worker): drain a finished one and, on the eval
            # cadence when idle, launch a new one - both non-blocking. Logged at eval_iter.
            finished = poll_eval(evaluator) if evaluator is not None else None
            if finished is not None:
                eval_iter, eval_c, eval_metrics = finished
                assert writer is not None
                for key, value in eval_metrics.items():
                    writer.add_scalar(key, value, eval_iter)
                writer.add_scalar("eval/ascension", eval_c, eval_iter)
                print(
                    f"  eval@{eval_iter} c={eval_c:.2f}: "
                    f"win_rate={eval_metrics[EVAL_WIN_RATE]:.3f} "
                    f"floor={eval_metrics[EVAL_FLOOR_AVG]:.2f} "
                    f"act2={eval_metrics[EVAL_ACT2_RATE]:.2f} "
                    f"reward={eval_metrics[EVAL_REWARD_AVG]:.4f}"
                )
                c_prev = c
                c, streak = step_curriculum(
                    c, streak, eval_metrics[EVAL_WIN_RATE], eval_c, curriculum
                )
                if c != c_prev:
                    frontier = frontier_level(c, curriculum.ascension_max)
                    print(f"  curriculum: c={c:.2f}, frontier A{frontier}")
            c = distributed.broadcast_float(c, rank_info.device)
            publish(handoff, model, iteration + 1, c)
            t_now = time.perf_counter()
            samples_per_s = steps_per_iteration / (t_now - t_iteration)
            t_iteration = t_now

            if evaluator is not None and iteration % cadence.eval_every == 0:
                submit_eval(evaluator, model, iteration, c)
            if iteration % cadence.log_every == 0:
                completed = distributed.gather_episodes(delivery.completed)
                if writer is not None:
                    print(
                        f"Iter {iteration} | steps={total_steps} | "
                        f"policy={metrics['loss/policy']:.4f} "
                        f"value={metrics['loss/value']:.4f} | episodes={len(completed)}"
                    )
                    scalars = {
                        **metrics,
                        "train/ascension": c,
                        "Steps/total": total_steps,
                        "Time/rollout": delivery.seconds_collect,
                        "Time/update": seconds_update,
                        "pipeline/actor_wait": delivery.seconds_wait,
                        "pipeline/learner_wait": seconds_learner_wait,
                        "pipeline/samples_per_s": samples_per_s,
                        **delivery.info,
                        **episode_scalars(completed),
                    }
                    if rank_info.device.type == "cuda":
                        peak = torch.cuda.max_memory_reserved(rank_info.device)
                        scalars["memory/learner_gb"] = peak / 2**30
                    for key, value in scalars.items():
                        writer.add_scalar(key, value, iteration)
            if rank_info.rank == 0 and iteration % cadence.save_every == 0:
                save_checkpoint(ckpt_path, model, optimizer, iteration, total_steps, c, lr_scale)
            if rank_info.rank == 0 and iteration > 0 and iteration % cadence.archive_every == 0:
                path = archive_path(exp_name, iteration)
                save_checkpoint(path, model, optimizer, iteration, total_steps, c, lr_scale)

            # A stop request on any rank stops every rank here, between collectives
            if distributed.any_rank(stop.is_set(), rank_info.device):
                if rank_info.rank == 0:
                    print("\nTraining interrupted. Saving checkpoint...")
                    save_checkpoint(
                        ckpt_path, model, optimizer, iteration, total_steps, c, lr_scale
                    )
                break
    finally:
        if handoff is not None:
            stop_actor(handoff, actor)
        if evaluator is not None:
            stop_eval(evaluator)
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    # SIGTERM or SIGINT on any rank stops every rank at the end of the current iteration,
    # after rank 0 checkpoints
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda signum, frame: stop.set())
    signal.signal(signal.SIGINT, lambda signum, frame: stop.set())
    rank_info = distributed.init()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path", nargs="?", default="src/rl/algorithms/actor_critic/config.yml"
    )
    parser.add_argument("--rollout-workers", type=int)
    args = parser.parse_args()
    config = load_config(args.config_path)

    # TF32 on Ampere+ GPUs: near-free matmul throughput (no-op on CPU/MPS)
    torch.set_float32_matmul_precision("high")

    # Every size its own symbol, process-wide: duck sizing ties the window blocks' widths that
    # match at the heads' first trace, and the heads recompile each time those widths part
    torch.fx.experimental._config.use_duck_shape = False

    # Set seeds
    seed = config["seed"]
    set_rng_seeds(seed)

    # Instance `ActorCritic` model and optimizer
    model = ActorCritic(**config["model"]).to(rank_info.device)
    optimizer = init_optimizer(config["optimizer"]["name"], model, **config["optimizer"]["kwargs"])
    if rank_info.rank == 0:
        make_experiment_dir(config["exp_name"], args.config_path)

    # Load checkpoint
    ckpt_path = checkpoint_path(config["exp_name"])
    start_iteration, total_steps, curriculum_c, lr_scale = load_checkpoint(
        ckpt_path, model, optimizer
    )

    # Offset the stream seeds so a resume doesn't replay the run's env-seed sequence
    # Every rank plays its own games; rank 0 keeps the single-GPU seeds
    if start_iteration > 0 or rank_info.rank > 0:
        set_rng_seeds(seed + start_iteration + 7919 * rank_info.rank)

    curriculum = AscensionCurriculum(**config["curriculum"])
    if curriculum_c is not None:
        curriculum = replace(curriculum, ascension_start=curriculum_c)
    rollout_config = RolloutConfig(**config["rollout"])
    if args.rollout_workers is not None:
        rollout_config = replace(rollout_config, rollout_workers=args.rollout_workers)
    if rank_info.rank == 0:
        if start_iteration > 0:
            print(f"Resuming from {ckpt_path}: iteration {start_iteration}, {total_steps} steps")
        print(f"Starting training: {config['exp_name']} on {rank_info.world_size} rank(s)")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params={n_params:,} ({n_params / 1e6:.2f}M)")
        print(
            f"  num_envs={rollout_config.num_envs}, "
            f"rollout_length={rollout_config.rollout_length}, "
            f"rollout_workers={rollout_config.rollout_workers}"
        )
    train(
        exp_name=config["exp_name"],
        num_iterations=config["num_iterations"],
        cadence=CadenceConfig(**config["cadence"]),
        model=model,
        model_config=config["model"],
        optimizer=optimizer,
        gamma=config["gamma"],
        curriculum=curriculum,
        ppo=PPOConfig(**config["ppo"]),
        rollout_config=rollout_config,
        rank_info=rank_info,
        stop=stop,
        start_iteration=start_iteration,
        total_steps=total_steps,
        lr_scale=lr_scale,
    )
    distributed.close()
