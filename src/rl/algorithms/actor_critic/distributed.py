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
    in the order rank 0's gradients arrived in it.
    """

    indices: list[list[int]]  # Each bucket's parameters, by index in `model.parameters()`
    x_grads: dict[int, torch.Tensor]  # The step's accumulated gradients, by parameter index
    works: list[dist.Work]  # The launched buckets' all-reduces, in bucket order
    x_flats: list[torch.Tensor]  # Their buffers: the gradients, then the reached flags
    arrival_order: list[int] | None  # The first step's arrival order; None once regrouped


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
    it has all of them. Without torchrun there are no buckets and no hooks.

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
        indices=[], x_grads={}, works=[], x_flats=[], arrival_order=[]
    )
    if not dist.is_initialized():
        return gradient_buckets

    params = list(model.parameters())
    trainable = [i for i in reversed(range(len(params))) if params[i].requires_grad]
    gradient_buckets.indices = _group_into_buckets(trainable, params)
    for i in trainable:
        # The hook gets the parameter as its argument: it holds only the buckets and the index
        params[i].register_post_accumulate_grad_hook(partial(_mark_ready, gradient_buckets, i))
    return gradient_buckets


def _group_into_buckets(order: list[int], params: list[nn.Parameter]) -> list[list[int]]:
    """Split the parameter indices in `order`, in that order, into buckets.

    Each bucket holds about `_BUCKET_BYTES` of gradients.

    Parameters
    ----------
    order : list[int]
        Indices into `params`, in the order the buckets take them.
    params : list[nn.Parameter]
        The model's parameters.

    Returns
    -------
    list[list[int]]
        Each bucket's parameters, by index into `params`.
    """
    indices: list[list[int]] = []
    num_bytes = 0
    for i in order:
        if not indices or num_bytes >= _BUCKET_BYTES:
            indices.append([])
            num_bytes = 0
        indices[-1].append(i)
        num_bytes += params[i].numel() * params[i].element_size()
    return indices


def _mark_ready(gradient_buckets: GradientBuckets, i: int, x_param: torch.Tensor) -> None:
    """Record parameter i's gradient, then launch every next bucket that has all its gradients.

    In the first step it also records the gradient's arrival. The buckets launch in bucket order,
    so every rank starts the same all-reduces in the same order.

    Parameters
    ----------
    gradient_buckets : GradientBuckets
        The step's buckets, updated in place.
    i : int
        The parameter's index in `model.parameters()`.
    x_param : torch.Tensor
        (the parameter's shape) float. The parameter, its gradient accumulated.
    """
    gradient_buckets.x_grads[i] = cast(torch.Tensor, x_param.grad)
    if gradient_buckets.arrival_order is not None:
        gradient_buckets.arrival_order.append(i)
    for indices_bucket in gradient_buckets.indices[len(gradient_buckets.works) :]:
        if not all(j in gradient_buckets.x_grads for j in indices_bucket):
            return
        x_grads_bucket = [gradient_buckets.x_grads[j] for j in indices_bucket]
        x_reached = torch.ones(
            len(indices_bucket), dtype=x_grads_bucket[0].dtype, device=x_grads_bucket[0].device
        )
        _launch_bucket(gradient_buckets, x_grads_bucket, x_reached)


def _launch_bucket(
    gradient_buckets: GradientBuckets, x_grads_bucket: list[torch.Tensor], x_reached: torch.Tensor
) -> None:
    """Start summing the next bucket's gradients and reached flags over the ranks, as one buffer.

    Parameters
    ----------
    gradient_buckets : GradientBuckets
        The step's buckets; the all-reduce and its buffer join them.
    x_grads_bucket : list[torch.Tensor]
        (its parameter's shape) float. Each of the bucket's gradients, in bucket order: zeros for
        a parameter this rank didn't reach.
    x_reached : torch.Tensor
        (len(x_grads_bucket),) float. 1 for each parameter this rank reached, 0 for the rest.
    """
    x_flat = torch.cat([x_grad.flatten() for x_grad in x_grads_bucket] + [x_reached])
    gradient_buckets.works.append(dist.all_reduce(x_flat, async_op=True))
    gradient_buckets.x_flats.append(x_flat)


def average_gradient_buckets(gradient_buckets: GradientBuckets, model: nn.Module) -> None:
    """Average the gradients over the ranks, bucket by bucket.

    Call it right after the backward, before any other collective: the ranks may have launched
    different numbers of buckets in it. It launches the buckets still waiting, with zeros and
    flag 0 for each parameter this rank didn't reach, then sets each gradient to its average over
    the ranks. A parameter no rank reached keeps no gradient, so Adam skips it as it does on one
    GPU. After the first step it regroups the buckets in the order rank 0's gradients arrived.

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
    for indices_bucket in gradient_buckets.indices[len(gradient_buckets.works) :]:
        # Ones, zeroed by fill_: a tensor from a list, or x[j] = 0.0, would make the CPU wait
        x_reached = torch.ones(
            len(indices_bucket),
            dtype=params[indices_bucket[0]].dtype,
            device=params[indices_bucket[0]].device,
        )
        for j, i in enumerate(indices_bucket):
            if i not in gradient_buckets.x_grads:
                gradient_buckets.x_grads[i] = torch.zeros_like(params[i])
                x_reached[j].fill_(0.0)
        x_grads_bucket = [gradient_buckets.x_grads[i] for i in indices_bucket]
        _launch_bucket(gradient_buckets, x_grads_bucket, x_reached)
    for work in gradient_buckets.works:
        work.wait()

    # One read of every bucket's flags: each .tolist() would make the CPU wait for the GPU
    reached = iter(
        torch.cat(
            [
                x_flat[-len(indices_bucket) :]
                for x_flat, indices_bucket in zip(
                    gradient_buckets.x_flats, gradient_buckets.indices
                )
            ]
        ).tolist()
    )
    for x_flat, indices_bucket in zip(gradient_buckets.x_flats, gradient_buckets.indices):
        x_averaged = x_flat[: -len(indices_bucket)] / dist.get_world_size()
        for i, x_grad in zip(
            indices_bucket, x_averaged.split([params[i].numel() for i in indices_bucket])
        ):
            params[i].grad = x_grad.view_as(params[i]) if next(reached) else None
    gradient_buckets.x_grads.clear()
    gradient_buckets.works.clear()
    gradient_buckets.x_flats.clear()

    # After the first step, every rank regroups its buckets in rank 0's arrival order
    if gradient_buckets.arrival_order is not None:
        arrived = set(gradient_buckets.arrival_order)
        unreached = [
            i
            for indices_bucket in gradient_buckets.indices
            for i in indices_bucket
            if i not in arrived
        ]
        x_order = torch.tensor(
            gradient_buckets.arrival_order + unreached, dtype=torch.long, device=params[0].device
        )
        dist.broadcast(x_order, src=0)
        gradient_buckets.indices = _group_into_buckets(x_order.tolist(), params)
        gradient_buckets.arrival_order = None


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
