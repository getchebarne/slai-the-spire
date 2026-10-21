"""The run's experiment folder: its config copy, its checkpoint and the checkpoint's archives"""

import os
import shutil

import torch

from src.rl.models import ActorCritic


def experiment_dir(exp_name: str) -> str:
    return f"experiments/{exp_name}"


def checkpoint_path(exp_name: str) -> str:
    return f"{experiment_dir(exp_name)}/checkpoint.pth"


def archive_path(exp_name: str, iteration: int) -> str:
    return f"{experiment_dir(exp_name)}/checkpoint-{iteration}.pth"


def make_experiment_dir(exp_name: str, config_path: str) -> None:
    """Create the run's folder and copy its config there"""
    os.makedirs(experiment_dir(exp_name), exist_ok=True)
    shutil.copy(config_path, f"{experiment_dir(exp_name)}/config.yml")


def save_checkpoint(
    path: str,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    total_steps: int,
    curriculum_c: float,
    lr_scale: float,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "total_steps": total_steps,
            "curriculum_c": curriculum_c,
            "lr_scale": lr_scale,
        },
        path,
    )


def load_checkpoint(
    path: str, model: ActorCritic, optimizer: torch.optim.Optimizer
) -> tuple[int, int, float | None, float]:
    """Resume model/optimizer from `path` if it exists; returns
    (start_iteration, total_steps, curriculum_c, lr_scale), or (0, 0, None, 1.0)
    for a fresh run. `model` must already sit on its device: optimizer.load_state_dict puts
    each moment beside its parameter."""
    if not os.path.exists(path):
        return 0, 0, None, 1.0

    # On the CPU: load_state_dict moves the moments to the parameters, but not Adam's step counts
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_iteration = ckpt["iteration"] + 1
    total_steps = ckpt["total_steps"]
    return start_iteration, total_steps, ckpt["curriculum_c"], ckpt["lr_scale"]
