"""Data parallelism over GPU ranks launched by torchrun.

Every rank runs its own Actor and Learner on its own games; the Learners average gradients each
step, so their weights stay identical. Without torchrun every function is a no-op on rank 0 of 1.
"""

import os
from dataclasses import dataclass
from functools import partial
from typing import cast

import torch
import torch.distributed as dist
import torch.nn as nn

from src.rl.algorithms.actor_critic.envs import EpisodeResult


# About this many bytes of gradients per bucket, each bucket one all-reduce
_BUCKET_BYTES = 5 * 2**20


@dataclass
class RankInfo:
    rank: int
    world_size: int
    device: torch.device


@dataclass
class GradientBuckets:
    """The trainable parameters in buckets, one all-reduce each, and the step's progress.

    The buckets start in reverse parameter order. After the first step every rank regroups them
    in the order in which rank 0 reached the parameters. A bucket's buffer is allocated with the
    bucket, and every step's all-reduce reuses it. The `batch_*` fields hold one entry per
    bucket, indexed by `idx_bucket` in launch order.
    """

    batch_idx_params: list[list[int]]                    # Its parameters, by index in `model.parameters()`
    batch_work: list[dist.Work | None]                   # Its all-reduce this step; None unlaunched
    batch_x_grad_of_param: list[dict[int, torch.Tensor]] # (its parameter's shape) float
    batch_x_flat: list[torch.Tensor]                     # (its numel + its parameter count,) float

    idx_bucket_of_param: dict[int, int]  # Each trainable parameter's bucket
    order_reached: list[int] | None      # The first step's order reached; None once regrouped


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


def hook_gradient_buckets(model: nn.Module) -> GradientBuckets:
    """Group the trainable parameters into buckets, and hook each one into its bucket.

    A bucket holds about `_BUCKET_BYTES` of gradients, and the backward starts its all-reduce once
    it has all of them. Every backward through the model must be followed by
    `average_gradient_buckets`: another backward first, for an auxiliary loss or a probe, would
    start all-reduces nobody waits for and leave the step's state stale. Without torchrun there
    are no buckets and no hooks.

    Parameters
    ----------
    model : nn.Module
        This rank's model; each trainable parameter gets a hook.

    Returns
    -------
    GradientBuckets
        The buckets, with no step started; empty without torchrun.
    """
    gradient_buckets = GradientBuckets(
        batch_idx_params=[],
        batch_work=[],
        batch_x_grad_of_param=[],
        batch_x_flat=[],
        idx_bucket_of_param={},
        order_reached=[],
    )
    if not dist.is_initialized():
        return gradient_buckets

    # Get model parameters
    params = list(model.parameters())

    # Filter trainable indexes
    idx_param_train = [
        idx_param for idx_param in reversed(range(len(params))) if params[idx_param].requires_grad
    ]

    # Group grainable parameters into buckets
    _group_into_buckets(gradient_buckets, idx_param_train, params)

    # Register a hook that fires after all of a parameter's gradients have been accumulated
    for idx_param in idx_param_train:
        params[idx_param].register_post_accumulate_grad_hook(
            partial(_mark_reached, gradient_buckets, idx_param)
        )

    return gradient_buckets


def _group_into_buckets(
    gradient_buckets: GradientBuckets, idx_param_train: list[int], params: list[nn.Parameter]
) -> None:
    """Split the parameter indices in `idx_param_train`, in that order, into buckets.

    Each bucket holds about `_BUCKET_BYTES` of gradients, and gets the buffer its all-reduces
    reuse. It works in place, setting the `batch_*` fields and `idx_bucket_of_param`, because
    the hooks are bound to `gradient_buckets`.

    Parameters
    ----------
    gradient_buckets : GradientBuckets
        The buckets, their `batch_*` fields and `idx_bucket_of_param` replaced.
    idx_param_train : list[int]
        Indices into `params`, in the order the buckets take them.
    params : list[nn.Parameter]
        The model's parameters.
    """
    batch_idx_params: list[list[int]] = []
    num_bytes = 0
    for idx_param in idx_param_train:
        # Add bucket
        if not batch_idx_params or num_bytes >= _BUCKET_BYTES:
            batch_idx_params.append([])
            num_bytes = 0

        # Add parameter to current bucket
        batch_idx_params[-1].append(idx_param)

        # Accumulate bytes
        param = params[idx_param]
        num_bytes += param.numel() * param.element_size()

    # Make each bucket's buffer, in its parameters' dtype and on their device
    batch_x_flat = []
    for idx_params in batch_idx_params:
        param_first = params[idx_params[0]]
        num_elements = sum(params[idx_param].numel() for idx_param in idx_params) + len(idx_params)
        x_flat = torch.empty(num_elements, dtype=param_first.dtype, device=param_first.device)
        batch_x_flat.append(x_flat)

    gradient_buckets.batch_idx_params = batch_idx_params
    gradient_buckets.batch_x_grad_of_param = [{} for _ in batch_idx_params]
    gradient_buckets.batch_work = [None for _ in batch_idx_params]
    gradient_buckets.batch_x_flat = batch_x_flat
    gradient_buckets.idx_bucket_of_param = {
        idx_param: idx_bucket
        for idx_bucket, idx_params in enumerate(batch_idx_params)
        for idx_param in idx_params
    }


def _mark_reached(
    gradient_buckets: GradientBuckets, idx_param: int, x_param: torch.Tensor
) -> None:
    """Record a parameter's gradient, then launch every next bucket that has all its gradients.

    In the first step it also records the order the parameters are reached in. The buckets
    launch in bucket order, so every rank starts the same all-reduces in the same order.

    Parameters
    ----------
    gradient_buckets : GradientBuckets
        The step's buckets, updated in place.
    idx_param : int
        The parameter's index in `model.parameters()`.
    x_param : torch.Tensor
        (the parameter's shape) float. The parameter, its gradient accumulated.

    """
    # Store the parameter's gradient
    idx_bucket = gradient_buckets.idx_bucket_of_param[idx_param]
    x_grad_of_param = gradient_buckets.batch_x_grad_of_param[idx_bucket]
    x_grad_of_param[idx_param] = cast(torch.Tensor, x_param.grad)

    if gradient_buckets.order_reached is not None:
        gradient_buckets.order_reached.append(idx_param)

    for idx_bucket_next, idx_params in enumerate(gradient_buckets.batch_idx_params):
        if gradient_buckets.batch_work[idx_bucket_next] is not None:
            continue

        # Check if the bucket's parameter's gradients are ready
        x_grad_of_param = gradient_buckets.batch_x_grad_of_param[idx_bucket_next]
        bucket_ready = len(x_grad_of_param) == len(idx_params)
        if not bucket_ready:
            return

        # If they are, collect them and launch the bucket
        batch_x_grad = [x_grad_of_param[idx_param_bucket] for idx_param_bucket in idx_params]
        _launch_bucket(gradient_buckets, idx_bucket_next, batch_x_grad)


def _launch_bucket(
    gradient_buckets: GradientBuckets, idx_bucket: int, batch_x_grad: list[torch.Tensor]
) -> None:
    """Start summing a bucket's gradients and reached flags over the ranks, in its buffer.

    Parameters
    ----------
    gradient_buckets : GradientBuckets
        The buckets: the bucket's buffer overwritten and its all-reduce started.
    idx_bucket : int
        The bucket's index, in launch order.
    batch_x_grad : list[torch.Tensor]
        (its parameter's shape) float. Each of the bucket's gradients, in bucket order: zeros for
        a parameter this rank didn't reach.
    """
    device = batch_x_grad[0].device
    idx_params = gradient_buckets.batch_idx_params[idx_bucket]
    x_grad_of_param = gradient_buckets.batch_x_grad_of_param[idx_bucket]
    x_flat = gradient_buckets.batch_x_flat[idx_bucket]

    # Whether this rank reached each parameter, sent from pinned memory without blocking on CUDA
    batch_reached = [float(idx_param in x_grad_of_param) for idx_param in idx_params]
    x_reached = torch.tensor(batch_reached, pin_memory=device.type == "cuda").to(
        device, non_blocking=True
    )

    # x_flat[:-len(params)]: flattened gradients for each parameter
    # x_flat[-len(params):]: reached flag for each parameter
    torch.cat([x_grad.flatten() for x_grad in batch_x_grad] + [x_reached], out=x_flat)

    # All-reduce both gradients and reached flags (async) + append refs
    gradient_buckets.batch_work[idx_bucket] = dist.all_reduce(x_flat, async_op=True)


def average_gradient_buckets(gradient_buckets: GradientBuckets, model: nn.Module) -> None:
    """Average the gradients over the ranks, bucket by bucket.

    Call it right after the backward, before any other collective: the ranks may have launched
    different numbers of buckets in it. It launches the buckets still waiting, with zeros and
    flag 0 for each parameter this rank didn't reach, then sets each gradient to its average over
    the ranks. A parameter no rank reached keeps no gradient, so Adam skips it as it does on one
    GPU. After the first step it regroups the buckets in the order in which rank 0 reached the
    parameters.

    Parameters
    ----------
    gradient_buckets : GradientBuckets
        The step's buckets, emptied for the next step.
    model : nn.Module
        This rank's model, each parameter's gradient replaced by its average over the ranks.
    """
    if not dist.is_initialized():
        return

    params = list(model.parameters())

    # The buckets still waiting, with zeros for each parameter this rank didn't reach
    for idx_bucket, idx_params in enumerate(gradient_buckets.batch_idx_params):
        if gradient_buckets.batch_work[idx_bucket] is not None:
            continue

        x_grad_of_param = gradient_buckets.batch_x_grad_of_param[idx_bucket]
        batch_x_grad = [
            x_grad_of_param[idx_param]
            if idx_param in x_grad_of_param
            else torch.zeros_like(params[idx_param])
            for idx_param in idx_params
        ]
        _launch_bucket(gradient_buckets, idx_bucket, batch_x_grad)

    # Wait for reductions
    for work in gradient_buckets.batch_work:
        cast(dist.Work, work).wait()

    # The trainable parameters' indices, in bucket order
    idx_param_train = [
        idx_param for idx_params in gradient_buckets.batch_idx_params for idx_param in idx_params
    ]

    # A parameter this rank reached counts as reached on every rank: read the sums only for a miss
    idxs_unreached_any: set[int] = set()
    if not all(
        len(gradient_buckets.batch_x_grad_of_param[idx_bucket]) == len(idx_params)
        for idx_bucket, idx_params in enumerate(gradient_buckets.batch_idx_params)
    ):
        # One read of every bucket's flags: each .tolist() would make the CPU wait for the GPU
        batch_reached_any = torch.cat(
            [
                gradient_buckets.batch_x_flat[idx_bucket][-len(idx_params) :]
                for idx_bucket, idx_params in enumerate(gradient_buckets.batch_idx_params)
            ]
        ).tolist()  # (N_param,)

        # The flags follow the parameters' bucket order
        idxs_unreached_any = {
            idx_param
            for idx_param, reached_any in zip(idx_param_train, batch_reached_any)
            if not reached_any
        }

    # Each bucket's averaged gradients, None for a parameter no rank reached
    for idx_bucket, idx_params in enumerate(gradient_buckets.batch_idx_params):
        x_flat = gradient_buckets.batch_x_flat[idx_bucket]
        x_averaged = x_flat[: -len(idx_params)] / dist.get_world_size()
        for idx_param, x_grad in zip(
            idx_params, x_averaged.split([params[idx_param].numel() for idx_param in idx_params])
        ):
            params[idx_param].grad = (
                None if idx_param in idxs_unreached_any else x_grad.view_as(params[idx_param])
            )

        # Reset the bucket for the next step
        gradient_buckets.batch_x_grad_of_param[idx_bucket].clear()
        gradient_buckets.batch_work[idx_bucket] = None

    # After the first step, every rank regroups its buckets in rank 0's order reached
    if gradient_buckets.order_reached is not None:
        idxs_reached = set(gradient_buckets.order_reached)
        idxs_unreached = [
            idx_param for idx_param in idx_param_train if idx_param not in idxs_reached
        ]
        x_idx_param = torch.tensor(
            gradient_buckets.order_reached + idxs_unreached,
            dtype=torch.long,
            device=params[0].device,
        )
        dist.broadcast(x_idx_param, src=0)
        _group_into_buckets(gradient_buckets, x_idx_param.tolist(), params)
        gradient_buckets.order_reached = None


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
    
    # Start reducing
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


def mean_over_ranks_on_device(x_value: torch.Tensor) -> torch.Tensor:
    """Average a tensor over the ranks, on its device: nothing is read back to the host.

    Parameters
    ----------
    x_value : torch.Tensor
        (...). This rank's tensor, on its device, where the all-reduce runs.

    Returns
    -------
    torch.Tensor
        (...). The mean of every rank's `x_value`, the same on every rank; `x_value` itself
        without torchrun.
    """
    if not dist.is_initialized():
        return x_value

    x_value_sum = x_value.detach().clone()
    dist.all_reduce(x_value_sum)

    return x_value_sum / dist.get_world_size()


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
