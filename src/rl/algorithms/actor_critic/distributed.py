"""Data parallelism over GPU ranks launched by torchrun.

Every rank runs its own Actor and Learner on its own games; the Learners average gradients each
step, so their weights stay identical. Without torchrun every function is a no-op on rank 0 of 1.
"""

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn

from src.rl.algorithms.actor_critic.envs import EpisodeResult


@dataclass
class RankInfo:
    rank: int
    world_size: int
    device: torch.device


@dataclass
class RankMean:
    """A scalar's mean over the ranks on its way to the host, as `mean_over_ranks` starts it."""

    x_value_sum_cpu: torch.Tensor   # () float64, on the host: every rank's value summed
    num_ranks: int                  # The number of ranks the sum is over
    event: torch.cuda.Event | None  # The sum's copy to the host on CUDA; None on the CPU


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
    device = params[0].device

    # Whether this rank reached each parameter, sent from pinned memory without blocking on CUDA
    batch_reached = [float(x_param.grad is not None) for x_param in params]
    x_reached = torch.tensor(batch_reached, pin_memory=device.type == "cuda").to(
        device, non_blocking=True
    )
    x_flat = torch.cat(
        [
            (x_param.grad if x_param.grad is not None else torch.zeros_like(x_param)).flatten()
            for x_param in params
        ]
        + [x_reached]
    )
    dist.all_reduce(x_flat)
    x_grad = x_flat[: -len(params)] / dist.get_world_size()

    # A parameter this rank reached counts as reached on every rank: read the sums only for a miss
    batch_reached_any = batch_reached if all(batch_reached) else x_flat[-len(params) :].tolist()
    offset = 0
    for x_param, reached in zip(params, batch_reached_any):
        x_param.grad = (
            x_grad[offset : offset + x_param.numel()].view_as(x_param) if reached else None
        )
        offset += x_param.numel()


def mean_over_ranks(x_value: torch.Tensor) -> RankMean:
    """Start averaging a scalar over the ranks, for `read_rank_mean` to read.

    The mean is the same number on every rank. On CUDA the all-reduce and the sum's copy to the
    host are waited for on a side stream: neither the host nor the compute stream waits for them;
    `read_rank_mean` waits for the copy.

    Parameters
    ----------
    x_value : torch.Tensor
        (). This rank's value, on its device, where the all-reduce runs.

    Returns
    -------
    RankMean
        The sum of every rank's `x_value` on its way to the host, and the number of ranks.
    """
    x_value_sum = x_value.detach().to(torch.float64)
    num_ranks = 1
    work = None
    if dist.is_initialized():
        work = dist.all_reduce(x_value_sum, async_op=True)
        num_ranks = dist.get_world_size()
    if x_value_sum.device.type != "cuda":
        if work is not None:
            work.wait()

        return RankMean(x_value_sum_cpu=x_value_sum, num_ranks=num_ranks, event=None)

    # A side stream waits for the value, then for the all-reduce, while the compute stream goes on
    stream = torch.cuda.Stream(device=x_value_sum.device)
    stream.wait_stream(torch.cuda.current_stream(x_value_sum.device))
    x_value_sum_cpu = torch.empty((), dtype=torch.float64, pin_memory=True)
    event = torch.cuda.Event()
    with torch.cuda.stream(stream):
        if work is not None:
            work.wait()

        # The sum into pinned memory, then an event that marks the copy's end
        x_value_sum_cpu.copy_(x_value_sum, non_blocking=True)
        event.record(stream)

    # The sum's memory is not reused before the side stream is done with it
    x_value_sum.record_stream(stream)

    return RankMean(x_value_sum_cpu=x_value_sum_cpu, num_ranks=num_ranks, event=event)


def read_rank_mean(rank_mean: RankMean) -> float:
    """Wait for a mean over the ranks to reach the host, and read it.

    Parameters
    ----------
    rank_mean : RankMean
        The mean, as `mean_over_ranks` started it.

    Returns
    -------
    float
        The mean of every rank's value.
    """
    if rank_mean.event is not None:
        rank_mean.event.synchronize()

    return rank_mean.x_value_sum_cpu.item() / rank_mean.num_ranks


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


def gather_episode_results(episode_results: list[EpisodeResult]) -> list[EpisodeResult]:
    """Gather every rank's finished episodes, on every rank.

    Parameters
    ----------
    episode_results : list[EpisodeResult]
        This rank's finished episodes.

    Returns
    -------
    list[EpisodeResult]
        Every rank's finished episodes, rank after rank.
    """
    if not dist.is_initialized():
        return episode_results
    gathered: list[list[EpisodeResult]] = [[] for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, episode_results)

    return [episode for rank_episode_results in gathered for episode in rank_episode_results]
