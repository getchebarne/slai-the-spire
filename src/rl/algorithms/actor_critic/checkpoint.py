"""The run's experiment folder: its config copy and its checkpoints."""

import glob
import os
import shutil
from dataclasses import dataclass

import torch

from src.rl.models import ActorCritic


@dataclass
class RunState:
    """What a checkpoint resumes besides the weights, the optimizer and the iteration."""

    total_steps: int          # Steps played so far, every rank's games counted
    ascension_current: float  # The soft-curriculum position
    streak: int               # The consecutive passes at ascension_current so far
    lr_scale: float           # The Adaptive-KL learning rate scale


def experiment_dir(exp_name: str) -> str:
    return f"experiments/{exp_name}"


def make_experiment_dir(exp_name: str, config_path: str) -> None:
    """Create the run's folder and its checkpoints folder, and copy its config there.

    Parameters
    ----------
    exp_name : str
        The run's name, which names its folder under experiments/.
    config_path : str
        The run's config file, copied into the folder as config.yml.
    """
    os.makedirs(f"{experiment_dir(exp_name)}/checkpoints", exist_ok=True)
    shutil.copy(config_path, f"{experiment_dir(exp_name)}/config.yml")


def save_checkpoint(
    exp_name: str,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    run_state: RunState,
    archive_every: int,
) -> None:
    """Save the run's state as checkpoints/iter-{iteration}.pth, then delete its older saves.

    The archives stay: the saves at a multiple of `archive_every` after iteration 0.

    Parameters
    ----------
    exp_name : str
        The run's name.
    model : ActorCritic
        The model.
    optimizer : torch.optim.Optimizer
        The model's optimizer.
    iteration : int
        The iteration just trained.
    run_state : RunState
        The run's progress.
    archive_every : int
        Every how many iterations a save stays as an archive.
    """
    exp_dir = experiment_dir(exp_name)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "total_steps": run_state.total_steps,
            "ascension_current": run_state.ascension_current,
            "streak": run_state.streak,
            "lr_scale": run_state.lr_scale,
        },
        f"{exp_dir}/checkpoints/iter-{iteration}.pth",
    )
    for saved in _saved_iterations(exp_dir):
        if saved != iteration and (saved == 0 or saved % archive_every != 0):
            os.remove(f"{exp_dir}/checkpoints/iter-{saved}.pth")


def load_checkpoint(
    exp_name: str, model: ActorCritic, optimizer: torch.optim.Optimizer
) -> tuple[int, RunState | None]:
    """Resume model/optimizer from the run's latest checkpoint, if it has one.

    `model` must already sit on its device: optimizer.load_state_dict puts each moment beside its
    parameter.

    Parameters
    ----------
    exp_name : str
        The run's name.
    model : ActorCritic
        The model, its weights loaded in place.
    optimizer : torch.optim.Optimizer
        The model's optimizer, its state loaded in place.

    Returns
    -------
    iteration_start : int
        The iteration after the checkpoint's, or 0 for a fresh run.
    run_state : RunState | None
        The checkpoint's progress, or None for a fresh run.
    """
    exp_dir = experiment_dir(exp_name)
    saved = _saved_iterations(exp_dir)
    if not saved:
        return 0, None

    # On the CPU: load_state_dict moves the moments to the parameters, but not Adam's step counts
    ckpt = torch.load(
        f"{exp_dir}/checkpoints/iter-{max(saved)}.pth", map_location="cpu", weights_only=True
    )
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    run_state = RunState(
        total_steps=ckpt["total_steps"],
        ascension_current=ckpt["ascension_current"],
        streak=ckpt["streak"],
        lr_scale=ckpt["lr_scale"],
    )
    return ckpt["iteration"] + 1, run_state


def _saved_iterations(exp_dir: str) -> list[int]:
    """Return the iterations whose checkpoint an experiment folder holds.

    Parameters
    ----------
    exp_dir : str
        The experiment folder.

    Returns
    -------
    list[int]
        The iterations of its checkpoints/iter-*.pth, in no order.
    """
    paths = glob.glob(f"{exp_dir}/checkpoints/iter-*.pth")
    return [int(os.path.basename(path)[len("iter-") : -len(".pth")]) for path in paths]
