"""Data parallelism over GPU ranks launched by torchrun. Every rank runs its own actor
and learner on its own games; the learners average gradients each step, so their
weights stay identical. Without torchrun every function is a no-op on rank 0 of 1"""

import os
from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn

from src.rl.algorithms.actor_critic.episode import EpisodeStats


# About this many bytes of gradients per bucket, each bucket one all-reduce
_BUCKET_BYTES = 5 * 2**20


@dataclass
class RankInfo:
    rank: int
    world_size: int
    device: torch.device


@dataclass
class GradientBuckets:
    """The trainable parameters in buckets, one all-reduce each, and the current step's progress
    through them"""

    # Each bucket's parameters, by index in model.parameters(): the last parameter first, then,
    # from the second step on, in the order rank 0's gradients arrived in the first
    indices: list[list[int]]
    x_grads: dict[int, torch.Tensor]  # the step's accumulated gradients, by parameter index
    works: list[dist.Work]  # the launched buckets' all-reduces, in bucket order
    x_flats: list[torch.Tensor]  # and their buffers: the gradients, then the reached flags

    # This rank's parameters, by index, in the order their gradients arrive in the first step;
    # None once the buckets follow rank 0's
    arrival_order: list[int] | None


def init() -> RankInfo:
    """This process's rank, the world size and the device it trains on"""
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
    """Start every rank from rank 0's weights"""
    if dist.is_initialized():
        for x in model.state_dict().values():
            dist.broadcast(x, src=0)


def hook_gradient_buckets(model: nn.Module) -> GradientBuckets:
    """Group the trainable parameters into buckets of about _BUCKET_BYTES of gradients, and hook
    each one so the backward all-reduces a bucket once it has all its gradients. Without
    torchrun, no buckets and no hooks"""
    buckets = GradientBuckets(indices=[], x_grads={}, works=[], x_flats=[], arrival_order=[])
    if not dist.is_initialized():
        return buckets
    params = list(model.parameters())
    trainable = [i for i in reversed(range(len(params))) if params[i].requires_grad]
    buckets.indices = _group_into_buckets(trainable, params)
    for i in trainable:
        # The hook gets the parameter as its argument: it holds only the buckets and the index
        params[i].register_post_accumulate_grad_hook(partial(_mark_ready, buckets, i))
    return buckets


def _group_into_buckets(order: list[int], params: list[nn.Parameter]) -> list[list[int]]:
    """The parameter indices in `order`, split in that order into buckets of about _BUCKET_BYTES
    of gradients"""
    indices: list[list[int]] = []
    num_bytes = 0
    for i in order:
        if not indices or num_bytes >= _BUCKET_BYTES:
            indices.append([])
            num_bytes = 0
        indices[-1].append(i)
        num_bytes += params[i].numel() * params[i].element_size()
    return indices


def _mark_ready(buckets: GradientBuckets, i: int, param: torch.Tensor) -> None:
    """Record parameter i's accumulated gradient, and its arrival in the first step, then launch
    each next bucket that has all its gradients, in bucket order: every rank starts the same
    all-reduces in the same order"""
    assert param.grad is not None
    buckets.x_grads[i] = param.grad
    if buckets.arrival_order is not None:
        buckets.arrival_order.append(i)
    for indices in buckets.indices[len(buckets.works) :]:
        if not all(j in buckets.x_grads for j in indices):
            return
        x_grads = [buckets.x_grads[j] for j in indices]
        _launch_bucket(buckets, x_grads, x_grads[0].new_ones(len(indices)))


def _launch_bucket(
    buckets: GradientBuckets, x_grads: list[torch.Tensor], x_reached: torch.Tensor
) -> None:
    """Start summing the next bucket's gradients and reached flags over the ranks, as one buffer"""
    x_flat = torch.cat([x_grad.flatten() for x_grad in x_grads] + [x_reached])
    buckets.works.append(dist.all_reduce(x_flat, async_op=True))
    buckets.x_flats.append(x_flat)


def average_gradient_buckets(buckets: GradientBuckets, model: nn.Module) -> None:
    """Right after the backward, before any other collective (the ranks may have launched
    different numbers of buckets in it): launch the buckets still waiting, with zeros and flag 0
    for each parameter this rank didn't reach, then set each gradient to its average over the
    ranks. A parameter no rank reached keeps no gradient, so Adam skips it as it does on one
    GPU. After the first step, regroup the buckets in the order rank 0's gradients arrived in it"""
    if not dist.is_initialized():
        return
    params = list(model.parameters())
    for indices in buckets.indices[len(buckets.works) :]:
        # Ones, zeroed by fill_: a tensor from a list, or x[j] = 0.0, would make the CPU wait
        x_reached = params[indices[0]].new_ones(len(indices))
        for j, i in enumerate(indices):
            if i not in buckets.x_grads:
                buckets.x_grads[i] = torch.zeros_like(params[i])
                x_reached[j].fill_(0.0)
        _launch_bucket(buckets, [buckets.x_grads[i] for i in indices], x_reached)
    for work in buckets.works:
        work.wait()

    # One read of every bucket's flags: each .tolist() would make the CPU wait for the GPU
    reached = iter(
        torch.cat(
            [x_flat[-len(indices) :] for x_flat, indices in zip(buckets.x_flats, buckets.indices)]
        ).tolist()
    )
    for x_flat, indices in zip(buckets.x_flats, buckets.indices):
        x_averaged = x_flat[: -len(indices)] / dist.get_world_size()
        for i, x_grad in zip(indices, x_averaged.split([params[i].numel() for i in indices])):
            params[i].grad = x_grad.view_as(params[i]) if next(reached) else None
    buckets.x_grads.clear()
    buckets.works.clear()
    buckets.x_flats.clear()

    # After the first step, every rank regroups its buckets in rank 0's order: the order its
    # gradients arrived in, then the parameters it didn't reach, in bucket order
    if buckets.arrival_order is not None:
        arrived = set(buckets.arrival_order)
        unreached = [i for indices in buckets.indices for i in indices if i not in arrived]
        x_order = torch.tensor(buckets.arrival_order + unreached, device=params[0].device)
        dist.broadcast(x_order, src=0)
        buckets.indices = _group_into_buckets(x_order.tolist(), params)
        buckets.arrival_order = None


def mean_over_ranks(value: float, device: torch.device) -> float:
    """`value` averaged over the ranks: the same number on every rank"""
    if not dist.is_initialized():
        return value
    x = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(x)
    return x.item() / dist.get_world_size()


def any_rank(flag: bool, device: torch.device) -> bool:
    """Whether `flag` is set on at least one rank"""
    if not dist.is_initialized():
        return flag
    x = torch.tensor([float(flag)], device=device)
    dist.all_reduce(x, op=dist.ReduceOp.MAX)
    return bool(x.item())


def broadcast_float(value: float, device: torch.device) -> float:
    """Rank 0's `value` on every rank"""
    if not dist.is_initialized():
        return value
    x = torch.tensor([value], dtype=torch.float64, device=device)
    dist.broadcast(x, src=0)
    return x.item()


def gather_episodes(episodes: list[EpisodeStats]) -> list[EpisodeStats]:
    """Every rank's finished episodes, on every rank"""
    if not dist.is_initialized():
        return episodes
    gathered: list[list[EpisodeStats]] = [[] for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, episodes)
    return [episode for rank_episodes in gathered for episode in rank_episodes]
