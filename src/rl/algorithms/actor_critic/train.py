"""The training run's entrypoint."""

import argparse
import signal
import threading
from dataclasses import replace

import torch

from src.rl.algorithms.actor_critic import distributed
from src.rl.algorithms.actor_critic.actor import RolloutConfig
from src.rl.algorithms.actor_critic.checkpoint import make_experiment_dir
from src.rl.algorithms.actor_critic.curriculum import AscensionCurriculum
from src.rl.algorithms.actor_critic.learner import CadenceConfig
from src.rl.algorithms.actor_critic.learner import train
from src.rl.algorithms.actor_critic.ppo import PPOConfig
from src.rl.utils import load_config


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

    # Every size its own symbol, process-wide: duck sizing ties sizes that match at the heads'
    # first trace, such as a block's 4 rows and the recorded actions' 4 columns, and the heads
    # recompile each time those sizes part
    torch.fx.experimental._config.use_duck_shape = False

    if rank_info.rank == 0:
        make_experiment_dir(config["exp_name"], args.config_path)
    rollout_config = RolloutConfig(**config["rollout"])
    if args.rollout_workers is not None:
        rollout_config = replace(rollout_config, rollout_workers=args.rollout_workers)
    train(
        exp_name=config["exp_name"],
        num_iterations=config["num_iterations"],
        seed=config["seed"],
        cadence_config=CadenceConfig(**config["cadence"]),
        model_config=config["model"],
        optimizer_config=config["optimizer"],
        gamma=config["gamma"],
        curriculum=AscensionCurriculum(**config["curriculum"]),
        ppo_config=PPOConfig(**config["ppo"]),
        rollout_config=rollout_config,
        rank_info=rank_info,
        stop=stop,
    )
    distributed.close()
