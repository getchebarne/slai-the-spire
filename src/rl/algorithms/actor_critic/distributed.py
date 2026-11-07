"""Data parallelism over GPU ranks launched by torchrun.

Every rank runs its own Actor and Learner on its own games; the Learners average gradients each
step, so their weights stay identical. Without torchrun every function is a no-op on rank 0 of 1.
"""

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn

from src.rl.algorithms.actor_critic.stats import EpisodeStats


@dataclass
class RankInfo:
    rank: int
    world_size: int
    device: torch.device


def init() -> RankInfo:
    """Start the process group under torchrun; return the rank, the world size and the device.

    Under torchrun, the local rank's GPU becomes the current device and the group runs NCCL on it,
    or Gloo without a GPU. Without torchrun nothing starts, and the process is rank 0 of 1.

    Returns
    -------
    RankInfo
        The rank, the world size and the device it trains on: the local rank's GPU, or the CPU
        without one.
    """
    if "WORLD_SIZE" not in os.environ:
        return RankInfo(0, 1, torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    if torch.cuda.is_available():
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", device_id=device)
    else:
        device = torch.device("cpu")
        dist.init_process_group("gloo")
    return RankInfo(dist.get_rank(), dist.get_world_size(), device)


def close() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def broadcast_from_rank0(model: nn.Module) -> None:
    """Start every rank from rank 0's weights.

    Parameters
    ----------
    model : nn.Module
        This rank's model, its weights overwritten in place with rank 0's.
    """
    if dist.is_initialized():
        for x_weight in model.state_dict().values():
            dist.broadcast(x_weight, src=0)


def average_gradients(model: nn.Module) -> None:
    """Average the gradients over the ranks in one all-reduce.

    A parameter no rank reached keeps no gradient, so Adam skips it as it does on one GPU.

    Parameters
    ----------
    model : nn.Module
        This rank's model, each parameter's gradient replaced by its average over the ranks.
    """
    if not dist.is_initialized():
        return
    params = list(model.parameters())
    x_flat = torch.cat(
        [
            (x_param.grad if x_param.grad is not None else torch.zeros_like(x_param)).flatten()
            for x_param in params
        ]
        + [
            torch.tensor(
                [float(x_param.grad is not None) for x_param in params],
                device=params[0].device,
            )
        ]
    )
    dist.all_reduce(x_flat)
    x_grads = x_flat[: -len(params)] / dist.get_world_size()
    offset = 0
    for x_param, reached in zip(params, x_flat[-len(params) :].tolist()):
        x_param.grad = (
            x_grads[offset : offset + x_param.numel()].view_as(x_param) if reached else None
        )
        offset += x_param.numel()


def mean_over_ranks(value: float, device: torch.device) -> float:
    """Average `value` over the ranks: the same number on every rank.

    Parameters
    ----------
    value : float
        This rank's value.
    device : torch.device
        This rank's device, where the all-reduce runs.

    Returns
    -------
    float
        The mean of every rank's `value`.
    """
    if not dist.is_initialized():
        return value
    x_value = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(x_value)
    return x_value.item() / dist.get_world_size()


def any_rank(flag: bool, device: torch.device) -> bool:
    """Check whether `flag` is set on at least one rank.

    Parameters
    ----------
    flag : bool
        This rank's flag.
    device : torch.device
        This rank's device, where the all-reduce runs.

    Returns
    -------
    bool
        True if `flag` is set on at least one rank; the same on every rank.
    """
    if not dist.is_initialized():
        return flag
    x_flag = torch.tensor([float(flag)], device=device)
    dist.all_reduce(x_flag, op=dist.ReduceOp.MAX)
    return bool(x_flag.item())


def broadcast_float(value: float, device: torch.device) -> float:
    """Broadcast rank 0's `value` to every rank.

    Parameters
    ----------
    value : float
        This rank's value; only rank 0's is kept.
    device : torch.device
        This rank's device, where the broadcast runs.

    Returns
    -------
    float
        Rank 0's `value`.
    """
    if not dist.is_initialized():
        return value
    x_value = torch.tensor([value], dtype=torch.float64, device=device)
    dist.broadcast(x_value, src=0)
    return x_value.item()


def gather_episodes(episodes: list[EpisodeStats]) -> list[EpisodeStats]:
    """Gather every rank's finished episodes, on every rank.

    Parameters
    ----------
    episodes : list[EpisodeStats]
        This rank's finished episodes.

    Returns
    -------
    list[EpisodeStats]
        Every rank's finished episodes, rank after rank.
    """
    if not dist.is_initialized():
        return episodes
    gathered: list[list[EpisodeStats]] = [[] for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, episodes)
    return [episode for rank_episodes in gathered for episode in rank_episodes]
