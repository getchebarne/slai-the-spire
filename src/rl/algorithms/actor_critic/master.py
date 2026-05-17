"""
PPO training with fixed-length rollouts and in-process environments.

No multiprocessing — all game environments run in the main process.
Each iteration collects a fixed number of steps from all environments,
auto-resetting games that end. GAE handles episode boundaries correctly.

The trainer holds `slai.GameEnv` instances directly (the previous
two-step card-targeting `EnvWrapper` was deleted with the inline-target
refactor: HeadCardPlay now picks card+target in a single forward pass).
"""

import os
import random
import shutil
from dataclasses import dataclass
from dataclasses import fields
from typing import Callable
from typing import Iterator

import slai
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.rl.action_space.masks import MaskBatch
from src.rl.action_space.masks import SELECTION_SIZES
from src.rl.action_space.masks import get_mask_batch
from src.rl.action_space.types import HeadTypePrimary
from src.rl.action_space.types import IS_DECISION_PRIMARY
from src.rl.action_space.types import NUM_PRIMARY_HEADS
from src.rl.action_space.types import PRIMARY_NUM_CHOICES
from src.rl.action_space.types import to_action
from src.rl.constants import ASCENSION_LEVEL
from src.rl.constants import MAX_MONSTERS
from src.rl.encoding.state import XGameState
from src.rl.encoding.state import encode_batch_view_game_state
from src.rl.models import ActorCritic
from src.rl.models.actor_critic import iter_primary_groups
from src.rl.models.heads import recompute_grouped_log_prob_and_entropy
from src.rl.reward import compute_reward
from src.rl.utils import init_optimizer
from src.rl.utils import load_config


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class Transition:
    """Single recorded transition. Carries enough state for PPO recompute
    without needing the original `slai.GameState` view."""

    x_game_state: XGameState  # Pre-encoded state (batch=1)
    head_type_primary: int
    primary_mask: torch.Tensor | None  # (1, num_choices) or None
    selection_mask: torch.Tensor  # (1, max_entities)
    primary_index: int  # -1 for direct primaries
    primary_log_prob: torch.Tensor
    selection_index: int  # -1 if terminal
    selection_log_prob: torch.Tensor
    # Inline target (for COMBAT_DEFAULT play-card-with-target):
    target_index: int  # -1 if not a targeting card. Doubles as "did we
                       # target?" gate for PPO recompute (no need to also
                       # store the target_required mask — if target_index
                       # < 0, no target was sampled, so nothing to replay).
    target_log_prob: torch.Tensor
    monster_alive_mask: torch.Tensor  # (1, MAX_MONSTERS) bool
    # Multi-pick retain (for COMBAT_AWAIT_RETAIN):
    retain_indices: torch.Tensor  # (1, MAX_SIZE_HAND) int64
    retain_log_prob: torch.Tensor
    retain_num: int
    # Per-pile group ids for grouped sampling at PPO recompute
    hand_group_ids: torch.Tensor  # (1, MAX_SIZE_HAND) int
    deck_group_ids: torch.Tensor  # (1, MAX_SIZE_DECK) int
    card_reward_group_ids: torch.Tensor  # (1, MAX_SIZE_COMBAT_CARD_REWARD) int
    value: torch.Tensor
    reward: float
    done: bool


@dataclass
class TrajectoryBatch:
    """Batched data for PPO training."""

    x_game_states: list[XGameState]
    head_type_primaries: torch.Tensor
    primary_masks: list[torch.Tensor | None]
    selection_masks: list[torch.Tensor]
    primary_indices: torch.Tensor
    primary_log_probs: torch.Tensor
    selection_indices: torch.Tensor
    selection_log_probs: torch.Tensor
    target_indices: torch.Tensor  # (N,)
    target_log_probs: torch.Tensor
    monster_alive_mask: torch.Tensor  # (N, MAX_MONSTERS)
    retain_indices: torch.Tensor  # (N, MAX_SIZE_HAND)
    retain_log_probs: torch.Tensor
    retain_nums: torch.Tensor  # (N,)
    hand_group_ids: torch.Tensor  # (N, MAX_SIZE_HAND)
    deck_group_ids: torch.Tensor  # (N, MAX_SIZE_DECK)
    card_reward_group_ids: torch.Tensor  # (N, MAX_SIZE_COMBAT_CARD_REWARD)
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor

    def __len__(self) -> int:
        return len(self.x_game_states)


@dataclass
class EpisodeStats:
    """Stats from a completed episode (for logging)."""

    total_reward: float
    length: int


# =============================================================================
# Environment Manager
# =============================================================================


class EnvironmentManager:
    """Holds N `slai.GameEnv` instances in-process (no IPC). Auto-resets
    on terminal."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self._envs: list[slai.GameEnv] = []
        self._obs: list[slai.GameState] = []
        for _ in range(num_envs):
            env, obs = self._make_env()
            self._envs.append(env)
            self._obs.append(obs)

    @staticmethod
    def _make_env() -> tuple[slai.GameEnv, slai.GameState]:
        env = slai.GameEnv(ascension=ASCENSION_LEVEL)
        obs = env.reset(seed=random.randint(0, 2**31 - 1))
        return env, obs

    def get_view_states(self) -> list[slai.GameState]:
        return self._obs

    def step(self, env_idx: int, action) -> tuple[float, bool]:
        """Apply action; return (reward, terminated). Auto-reset on terminal."""
        prev = self._obs[env_idx]
        nxt, _engine_reward, terminated, _truncated, _info = self._envs[env_idx].step(action)
        reward = compute_reward(prev, nxt, terminated)
        if terminated:
            env, nxt = self._make_env()
            self._envs[env_idx] = env
        self._obs[env_idx] = nxt
        return reward, terminated


# =============================================================================
# XGameState Helpers
# =============================================================================


def _map_xgs(x: XGameState, op: Callable[[torch.Tensor], torch.Tensor]) -> XGameState:
    return XGameState(**{f.name: op(getattr(x, f.name)) for f in fields(XGameState)})


def _move_x_game_state(x: XGameState, device: torch.device) -> XGameState:
    return _map_xgs(x, lambda t: t.to(device))


def _concat_x_game_states(x_game_states: list[XGameState]) -> XGameState:
    return XGameState(
        **{
            f.name: torch.cat([getattr(x, f.name) for x in x_game_states], dim=0)
            for f in fields(XGameState)
        }
    )


def _slice_x_game_state(x_game_state: XGameState, idx: int) -> XGameState:
    return _map_xgs(x_game_state, lambda t: t[idx : idx + 1])


# =============================================================================
# Mask Helpers
# =============================================================================


def _build_route_map(mask_batch: MaskBatch) -> dict[tuple[int, int], int]:
    route_map: dict[tuple[int, int], int] = {}
    for htp in range(NUM_PRIMARY_HEADS):
        route = mask_batch.route[htp]
        if len(route) == 0:
            continue
        for local_idx, batch_idx in enumerate(route.cpu().tolist()):
            route_map[(htp, batch_idx)] = local_idx
    return route_map


def _extract_per_sample_masks(
    mask_batch: MaskBatch,
    batch_idx: int,
    htp: int,
    route_map: dict[tuple[int, int], int],
) -> tuple[torch.Tensor | None, torch.Tensor]:
    local_idx = route_map[(htp, batch_idx)]
    primary_mask = None
    if IS_DECISION_PRIMARY[htp]:
        primary_mask = mask_batch.primary_masks[htp][local_idx : local_idx + 1]
    selection_mask = mask_batch.selection_masks[htp][local_idx : local_idx + 1]
    return primary_mask, selection_mask


def _build_mask_batch_from_samples(
    head_type_primaries: list[int],
    primary_masks: list[torch.Tensor | None],
    selection_masks: list[torch.Tensor],
    monster_alive_mask: torch.Tensor,  # (B, MAX_MONSTERS)
    retain_nums: torch.Tensor,  # (B,)
    hand_group_ids: torch.Tensor,  # (B, MAX_SIZE_HAND)
    deck_group_ids: torch.Tensor,  # (B, MAX_SIZE_DECK)
    card_reward_group_ids: torch.Tensor,  # (B, MAX_SIZE_COMBAT_CARD_REWARD)
    device: torch.device,
) -> MaskBatch:
    """Rebuild MaskBatch from per-sample data for PPO recomputation.

    `target_required` is None: PPO recompute uses the recorded
    `target_index >= 0` to gate target replay and never reads it."""
    route_lists: list[list[int]] = [[] for _ in range(NUM_PRIMARY_HEADS)]
    for i, htp in enumerate(head_type_primaries):
        route_lists[htp].append(i)

    route: list[torch.Tensor] = [None] * NUM_PRIMARY_HEADS  # type: ignore
    pm_list: list[torch.Tensor] = [None] * NUM_PRIMARY_HEADS  # type: ignore
    sm_list: list[torch.Tensor] = [None] * NUM_PRIMARY_HEADS  # type: ignore

    for htp in range(NUM_PRIMARY_HEADS):
        idxs = route_lists[htp]
        route[htp] = torch.tensor(idxs, dtype=torch.long, device=device)

        if not idxs:
            if IS_DECISION_PRIMARY[htp]:
                pm_list[htp] = torch.zeros(
                    0, PRIMARY_NUM_CHOICES[htp], dtype=torch.bool, device=device
                )
            else:
                pm_list[htp] = torch.empty(0, dtype=torch.bool, device=device)
            sm_list[htp] = torch.zeros(0, SELECTION_SIZES[htp], dtype=torch.bool, device=device)
            continue

        if IS_DECISION_PRIMARY[htp]:
            pm_list[htp] = torch.cat([primary_masks[i] for i in idxs], dim=0).to(device)
        else:
            pm_list[htp] = torch.empty(0, dtype=torch.bool, device=device)

        sm_list[htp] = torch.cat([selection_masks[i] for i in idxs], dim=0).to(device)

    return MaskBatch(
        route=route,
        primary_masks=pm_list,
        selection_masks=sm_list,
        target_required=None,
        monster_alive_mask=monster_alive_mask.to(device),
        retain_nums=retain_nums.to(device),
        hand_group_ids=hand_group_ids.to(device),
        deck_group_ids=deck_group_ids.to(device),
        card_reward_group_ids=card_reward_group_ids.to(device),
    )


# =============================================================================
# Rollout Collection
# =============================================================================


def _collect_rollout(
    model: ActorCritic,
    env_mgr: EnvironmentManager,
    rollout_length: int,
    device: torch.device,
) -> tuple[list[list[Transition]], list[torch.Tensor], list[EpisodeStats]]:
    num_envs = env_mgr.num_envs
    transitions: list[list[Transition]] = [[] for _ in range(num_envs)]
    completed_episodes: list[EpisodeStats] = []

    ep_rewards = [0.0] * num_envs
    ep_lengths = [0] * num_envs

    model.eval()
    with torch.no_grad():
        for _ in range(rollout_length):
            view_states = env_mgr.get_view_states()
            x_game_state = encode_batch_view_game_state(view_states, device)
            mask_batch = get_mask_batch(view_states, device)
            output = model(x_game_state, mask_batch, sample=True)

            htps = output.head_type_primaries.cpu().tolist()
            pis = output.primary_indices.cpu().tolist()
            sis = output.selection_indices.cpu().tolist()
            tis = output.target_indices.cpu().tolist()
            ris_full = output.retain_indices.cpu()  # (B, MAX_SIZE_HAND) — keep tensor

            route_map = _build_route_map(mask_batch)

            for i in range(num_envs):
                htp = htps[i]
                ri_list = [int(x) for x in ris_full[i].tolist() if x >= 0]
                action = to_action(
                    HeadTypePrimary(htp), pis[i], sis[i],
                    target_index=tis[i],
                    retain_indices=ri_list,
                )
                primary_mask, selection_mask = _extract_per_sample_masks(
                    mask_batch, i, htp, route_map
                )

                reward, done = env_mgr.step(i, action)

                transitions[i].append(
                    Transition(
                        x_game_state=_slice_x_game_state(x_game_state, i),
                        head_type_primary=htp,
                        primary_mask=primary_mask,
                        selection_mask=selection_mask,
                        primary_index=pis[i],
                        primary_log_prob=output.primary_log_probs[i],
                        selection_index=sis[i],
                        selection_log_prob=output.selection_log_probs[i],
                        target_index=tis[i],
                        target_log_prob=output.target_log_probs[i],
                        monster_alive_mask=mask_batch.monster_alive_mask[i : i + 1],
                        retain_indices=output.retain_indices[i : i + 1],
                        retain_log_prob=output.retain_log_probs[i],
                        retain_num=int(mask_batch.retain_nums[i].item()),
                        hand_group_ids=mask_batch.hand_group_ids[i : i + 1],
                        deck_group_ids=mask_batch.deck_group_ids[i : i + 1],
                        card_reward_group_ids=mask_batch.card_reward_group_ids[i : i + 1],
                        value=output.values[i],
                        reward=reward,
                        done=done,
                    )
                )

                ep_rewards[i] += reward
                ep_lengths[i] += 1
                if done:
                    completed_episodes.append(EpisodeStats(ep_rewards[i], ep_lengths[i]))
                    ep_rewards[i] = 0.0
                    ep_lengths[i] = 0

        # Bootstrap values for GAE at truncation point
        view_states = env_mgr.get_view_states()
        x_game_state = encode_batch_view_game_state(view_states, device)
        mask_batch = get_mask_batch(view_states, device)
        output = model(x_game_state, mask_batch, sample=False)
        bootstrap_values = [output.values[i] for i in range(num_envs)]

    model.train()
    return transitions, bootstrap_values, completed_episodes


def _run_eval_episode(
    model: ActorCritic,
    device: torch.device,
) -> tuple[float, int]:
    """Run a single greedy evaluation episode."""
    env = slai.GameEnv(ascension=ASCENSION_LEVEL)
    obs = env.reset(seed=random.randint(0, 2**31 - 1))

    total_reward = 0.0
    length = 0
    terminated = False

    model.eval()
    with torch.no_grad():
        while not terminated:
            x = encode_batch_view_game_state([obs], device)
            mb = get_mask_batch([obs], device)
            output = model(x, mb, sample=False)
            action = output.get_action(0)

            prev = obs
            obs, terminated = env.step(action)
            reward = compute_reward(prev, obs, terminated)

            total_reward += reward
            length += 1

    model.train()
    return total_reward, length


# =============================================================================
# GAE and Batch Creation
# =============================================================================


def _compute_gae(
    rewards: list[float],
    values: list[torch.Tensor],
    dones: list[bool],
    bootstrap_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[list[float], list[float]]:
    T = len(rewards)
    advantages = [0.0] * T
    returns = [0.0] * T

    values_cpu = torch.stack(values).squeeze(-1).cpu().tolist()
    bootstrap_cpu = bootstrap_value.cpu().item()

    gae = 0.0
    for t in reversed(range(T)):
        next_value = bootstrap_cpu if t == T - 1 else values_cpu[t + 1]
        non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * non_terminal - values_cpu[t]
        gae = delta + gamma * lam * non_terminal * gae
        advantages[t] = gae
        returns[t] = gae + values_cpu[t]

    return returns, advantages


def _create_batch(
    transitions: list[list[Transition]],
    bootstrap_values: list[torch.Tensor],
    gamma: float,
    lam: float,
    device: torch.device,
) -> TrajectoryBatch:
    all_x_game_states = []
    all_head_type_primaries = []
    all_primary_masks = []
    all_selection_masks = []
    all_primary_indices = []
    all_primary_log_probs = []
    all_selection_indices = []
    all_selection_log_probs = []
    all_target_indices = []
    all_target_log_probs = []
    all_monster_alive_mask = []
    all_retain_indices = []
    all_retain_log_probs = []
    all_retain_nums = []
    all_hand_group_ids = []
    all_deck_group_ids = []
    all_card_reward_group_ids = []
    all_values = []
    all_returns = []
    all_advantages = []

    for env_idx, env_transitions in enumerate(transitions):
        if not env_transitions:
            continue
        rewards = [t.reward for t in env_transitions]
        values = [t.value for t in env_transitions]
        dones = [t.done for t in env_transitions]

        ret, adv = _compute_gae(
            rewards, values, dones, bootstrap_values[env_idx], gamma, lam
        )

        for i, trans in enumerate(env_transitions):
            all_x_game_states.append(trans.x_game_state)
            all_head_type_primaries.append(trans.head_type_primary)
            all_primary_masks.append(trans.primary_mask)
            all_selection_masks.append(trans.selection_mask)
            all_primary_indices.append(trans.primary_index)
            all_primary_log_probs.append(trans.primary_log_prob)
            all_selection_indices.append(trans.selection_index)
            all_selection_log_probs.append(trans.selection_log_prob)
            all_target_indices.append(trans.target_index)
            all_target_log_probs.append(trans.target_log_prob)
            all_monster_alive_mask.append(trans.monster_alive_mask)
            all_retain_indices.append(trans.retain_indices)
            all_retain_log_probs.append(trans.retain_log_prob)
            all_retain_nums.append(trans.retain_num)
            all_hand_group_ids.append(trans.hand_group_ids)
            all_deck_group_ids.append(trans.deck_group_ids)
            all_card_reward_group_ids.append(trans.card_reward_group_ids)
            all_values.append(trans.value)
            all_returns.append(ret[i])
            all_advantages.append(adv[i])

    batch = TrajectoryBatch(
        x_game_states=all_x_game_states,
        head_type_primaries=torch.tensor(all_head_type_primaries, dtype=torch.long, device=device),
        primary_masks=all_primary_masks,
        selection_masks=all_selection_masks,
        primary_indices=torch.tensor(all_primary_indices, dtype=torch.long, device=device),
        primary_log_probs=torch.stack(all_primary_log_probs).detach().to(device),
        selection_indices=torch.tensor(all_selection_indices, dtype=torch.long, device=device),
        selection_log_probs=torch.stack(all_selection_log_probs).detach().to(device),
        target_indices=torch.tensor(all_target_indices, dtype=torch.long, device=device),
        target_log_probs=torch.stack(all_target_log_probs).detach().to(device),
        monster_alive_mask=torch.cat(all_monster_alive_mask, dim=0).to(device),
        retain_indices=torch.cat(all_retain_indices, dim=0).to(device),
        retain_log_probs=torch.stack(all_retain_log_probs).detach().to(device),
        retain_nums=torch.tensor(all_retain_nums, dtype=torch.long, device=device),
        hand_group_ids=torch.cat(all_hand_group_ids, dim=0).to(device),
        deck_group_ids=torch.cat(all_deck_group_ids, dim=0).to(device),
        card_reward_group_ids=torch.cat(all_card_reward_group_ids, dim=0).to(device),
        values=torch.cat(all_values, dim=0).detach(),
        returns=torch.tensor(all_returns, dtype=torch.float32, device=device).view(-1, 1),
        advantages=torch.tensor(all_advantages, dtype=torch.float32, device=device).view(-1, 1),
    )

    # Advantage normalization. `torch.std` is NaN when N==1; skip
    # normalization in that degenerate case (otherwise NaN advantages
    # corrupt the policy gradient and the model goes NaN by epoch 2).
    if batch.advantages.numel() > 1:
        batch.advantages = (batch.advantages - torch.mean(batch.advantages)) / (
            torch.std(batch.advantages) + 1e-8
        )
    return batch


# =============================================================================
# PPO Update
# =============================================================================


def _minibatch_indices(total: int, minibatch_size: int) -> Iterator[list[int]]:
    indices = list(range(total))
    random.shuffle(indices)
    for i in range(0, total, minibatch_size):
        yield indices[i : i + minibatch_size]


def _recompute_log_probs_batch(
    model: ActorCritic,
    x_game_states: list[XGameState],
    head_type_primaries: list[int],
    primary_masks: list[torch.Tensor | None],
    selection_masks: list[torch.Tensor],
    monster_alive_mask: torch.Tensor,  # (B, MAX_MONSTERS)
    retain_nums: torch.Tensor,  # (B,)
    hand_group_ids: torch.Tensor,  # (B, MAX_SIZE_HAND)
    deck_group_ids: torch.Tensor,  # (B, MAX_SIZE_DECK)
    card_reward_group_ids: torch.Tensor,  # (B, MAX_SIZE_COMBAT_CARD_REWARD)
    primary_indices: torch.Tensor,
    selection_indices: torch.Tensor,
    target_indices: torch.Tensor,
    retain_indices: torch.Tensor,  # (B, MAX_SIZE_HAND)
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute log_probs and entropies under current policy for a minibatch.

    Returns (log_probs, entropies, values) — each (B,) or (B, 1).
    """
    B = len(x_game_states)

    x_game_state = _concat_x_game_states(x_game_states)
    x_game_state = _move_x_game_state(x_game_state, device)

    mask_batch = _build_mask_batch_from_samples(
        head_type_primaries, primary_masks, selection_masks,
        monster_alive_mask, retain_nums,
        hand_group_ids, deck_group_ids, card_reward_group_ids,
        device,
    )

    primary_indices = primary_indices.to(device)
    selection_indices = selection_indices.to(device)
    target_indices = target_indices.to(device)
    retain_indices = retain_indices.to(device)

    core_out = model.core(x_game_state)
    values = model.head_value(core_out.x_global)

    total_log_probs = torch.zeros(B, device=device)
    total_entropies = torch.zeros(B, device=device)

    for ctx in iter_primary_groups(core_out, mask_batch):
        htp = ctx.htp
        idx = ctx.idx

        multi_head = model._multi_pick_heads[htp]
        if multi_head is not None:
            nums = mask_batch.retain_nums[idx]
            recorded = retain_indices[idx]
            log_probs_group, entropy_group = multi_head.recompute_log_prob(
                ctx.entities_group, ctx.x_global_group, ctx.selection_mask, nums, recorded,
                group_ids=ctx.group_ids_group,
            )
            total_log_probs[idx] += log_probs_group
            total_entropies[idx] += entropy_group
            continue

        if IS_DECISION_PRIMARY[htp]:
            decision_head = model._decision_heads[htp]
            out = decision_head(ctx.x_global_group, ctx.primary_mask, sample=False)

            primary_dist = torch.distributions.Categorical(logits=out.logits)
            group_primary_idx = primary_indices[idx]
            total_log_probs[idx] += primary_dist.log_prob(group_primary_idx)
            total_entropies[idx] += primary_dist.entropy()

            needs_secondary = group_primary_idx == 1
            if torch.any(needs_secondary):
                sec_local = torch.nonzero(needs_secondary, as_tuple=True)[0]
                sec_batch = idx[sec_local]
                sec_x_global = ctx.x_global_group[sec_local]
                sec_mask = ctx.selection_mask[sec_local]
                sec_entities = (
                    ctx.entities_group[sec_local]
                    if ctx.entities_group is not None
                    else None
                )
                sec_group_ids = (
                    ctx.group_ids_group[sec_local]
                    if ctx.group_ids_group is not None
                    else None
                )
                sec_indices = selection_indices[sec_batch]

                sel_head = model._selection_heads[htp]
                sec_out = sel_head(
                    sec_entities, sec_x_global, sec_mask, sample=False,
                    group_ids=sec_group_ids,
                )

                sec_log_probs, sec_entropy = recompute_grouped_log_prob_and_entropy(
                    sec_out.logits, sec_indices, group_ids=sec_group_ids,
                )
                total_log_probs[sec_batch] += sec_log_probs
                total_entropies[sec_batch] += sec_entropy

                if htp == HeadTypePrimary.COMBAT_DEFAULT:
                    sec_target_idx = target_indices[sec_batch]
                    has_target = sec_target_idx >= 0
                    if torch.any(has_target):
                        tgt_local = torch.nonzero(has_target, as_tuple=True)[0]
                        tgt_batch = sec_batch[tgt_local]
                        tgt_card_idx = sec_indices[tgt_local]
                        tgt_recorded = sec_target_idx[tgt_local]

                        x_active_card = core_out.x_hand[tgt_batch, tgt_card_idx]
                        tgt_x_global = sec_x_global[tgt_local]
                        tgt_monsters = core_out.x_monsters[tgt_batch]
                        tgt_mask = mask_batch.monster_alive_mask[tgt_batch]

                        tgt_out = model.head_monster_select(
                            tgt_monsters, tgt_x_global, tgt_mask, sample=False,
                            x_active_card=x_active_card,
                        )
                        tgt_log_probs, tgt_entropy = recompute_grouped_log_prob_and_entropy(
                            tgt_out.logits, tgt_recorded,
                        )
                        total_log_probs[tgt_batch] += tgt_log_probs
                        total_entropies[tgt_batch] += tgt_entropy
        else:
            sel_indices = selection_indices[idx]
            sel_head = model._selection_heads[htp]
            sel_out = sel_head(
                ctx.entities_group, ctx.x_global_group, ctx.selection_mask, sample=False,
                group_ids=ctx.group_ids_group,
            )
            sel_log_probs, sel_entropy = recompute_grouped_log_prob_and_entropy(
                sel_out.logits, sel_indices, group_ids=ctx.group_ids_group,
            )
            total_log_probs[idx] += sel_log_probs
            total_entropies[idx] += sel_entropy

    return total_log_probs, total_entropies, values


def _update_ppo(
    model: ActorCritic,
    batch: TrajectoryBatch,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    minibatch_size: int,
    clip_eps: float,
    clip_value_loss: bool,
    coef_value: float,
    coef_entropy: float,
    max_grad_norm: float,
    device: torch.device,
) -> tuple[float, float, float]:
    total_loss_policy = 0.0
    total_loss_value = 0.0
    total_loss_entropy = 0.0
    num_updates = 0

    for _ in range(num_epochs):
        for mb_idxs in _minibatch_indices(len(batch), minibatch_size):
            mb_x_states = [batch.x_game_states[i] for i in mb_idxs]
            mb_htps = batch.head_type_primaries[mb_idxs].cpu().tolist()
            mb_primary_masks = [batch.primary_masks[i] for i in mb_idxs]
            mb_selection_masks = [batch.selection_masks[i] for i in mb_idxs]
            mb_primary_indices = batch.primary_indices[mb_idxs]
            mb_selection_indices = batch.selection_indices[mb_idxs]
            mb_target_indices = batch.target_indices[mb_idxs]
            mb_retain_indices = batch.retain_indices[mb_idxs]
            mb_monster_alive = batch.monster_alive_mask[mb_idxs]
            mb_retain_nums = batch.retain_nums[mb_idxs]
            mb_hand_group_ids = batch.hand_group_ids[mb_idxs]
            mb_deck_group_ids = batch.deck_group_ids[mb_idxs]
            mb_card_reward_group_ids = batch.card_reward_group_ids[mb_idxs]

            log_probs_new, entropies, values_new = _recompute_log_probs_batch(
                model,
                mb_x_states,
                mb_htps,
                mb_primary_masks,
                mb_selection_masks,
                mb_monster_alive,
                mb_retain_nums,
                mb_hand_group_ids,
                mb_deck_group_ids,
                mb_card_reward_group_ids,
                mb_primary_indices,
                mb_selection_indices,
                mb_target_indices,
                mb_retain_indices,
                device,
            )

            log_probs_old = (
                batch.primary_log_probs[mb_idxs].to(device)
                + batch.selection_log_probs[mb_idxs].to(device)
                + batch.target_log_probs[mb_idxs].to(device)
                + batch.retain_log_probs[mb_idxs].to(device)
            )

            advantages = torch.squeeze(batch.advantages[mb_idxs]).to(device)
            returns = batch.returns[mb_idxs].to(device)
            values_old = batch.values[mb_idxs].to(device)

            ratio = torch.exp(log_probs_new - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            loss_policy = -torch.mean(torch.min(surr1, surr2))

            if clip_value_loss:
                values_clipped = values_old + torch.clamp(
                    values_new - values_old, -clip_eps, clip_eps
                )
                loss_value_unclipped = torch.pow(values_new - returns, 2)
                loss_value_clipped = torch.pow(values_clipped - returns, 2)
                loss_value = 0.5 * torch.mean(torch.max(loss_value_unclipped, loss_value_clipped))
            else:
                loss_value = F.mse_loss(values_new, returns)

            loss_entropy = -torch.mean(entropies)
            loss = loss_policy + coef_value * loss_value + coef_entropy * loss_entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss_policy += loss_policy.item()
            total_loss_value += loss_value.item()
            total_loss_entropy += loss_entropy.item()
            num_updates += 1

    return (
        total_loss_policy / num_updates,
        total_loss_value / num_updates,
        total_loss_entropy / num_updates,
    )


# =============================================================================
# Training Loop
# =============================================================================


def _get_entropy_schedule(
    num_iterations: int,
    elbow: int,
    max_coef: float,
    min_coef: float,
) -> list[float]:
    coefs = []
    slope = (min_coef - max_coef) / elbow
    for it in range(num_iterations):
        if it <= elbow:
            coefs.append(slope * it + max_coef)
        else:
            coefs.append(min_coef)
    return coefs


def train(
    exp_name: str,
    num_iterations: int,
    log_every: int,
    save_every: int,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout_length: int,
    num_epochs: int,
    minibatch_size: int,
    clip_eps: float,
    clip_value_loss: bool,
    gamma: float,
    lam: float,
    coef_value: float,
    coefs_entropy: list[float],
    max_grad_norm: float,
    num_envs: int,
    device: torch.device,
) -> None:
    writer = SummaryWriter(f"experiments/{exp_name}")
    model.to(device)

    env_mgr = EnvironmentManager(num_envs)
    total_steps = 0

    try:
        for iteration in range(num_iterations):
            coef_entropy = coefs_entropy[iteration]

            transitions, bootstrap_values, completed_episodes = _collect_rollout(
                model, env_mgr, rollout_length, device
            )

            total_steps += rollout_length * num_envs

            batch = _create_batch(transitions, bootstrap_values, gamma, lam, device)
            if len(batch) == 0:
                print(f"Iteration {iteration}: Empty batch, skipping")
                continue

            loss_policy, loss_value, loss_entropy = _update_ppo(
                model, batch, optimizer, num_epochs, minibatch_size,
                clip_eps, clip_value_loss, coef_value, coef_entropy,
                max_grad_norm, device,
            )

            if iteration % log_every == 0:
                print(
                    f"Iter {iteration} | steps={total_steps} | "
                    f"policy={loss_policy:.4f} value={loss_value:.4f} | "
                    f"episodes={len(completed_episodes)}"
                )
                writer.add_scalar("Loss/policy", loss_policy, iteration)
                writer.add_scalar("Loss/value", loss_value, iteration)
                writer.add_scalar("Loss/entropy", loss_entropy, iteration)
                writer.add_scalar("Entropy/coef", coef_entropy, iteration)
                writer.add_scalar("Steps/total", total_steps, iteration)

                if completed_episodes:
                    avg_reward = sum(e.total_reward for e in completed_episodes) / len(completed_episodes)
                    avg_length = sum(e.length for e in completed_episodes) / len(completed_episodes)
                    writer.add_scalar("Episode/avg_reward", avg_reward, iteration)
                    writer.add_scalar("Episode/avg_length", avg_length, iteration)
                    writer.add_scalar("Episode/completed_count", len(completed_episodes), iteration)

                eval_reward, eval_length = _run_eval_episode(model, device)
                print(f"  eval: reward={eval_reward:.4f}, length={eval_length}")
                writer.add_scalar("Eval/reward", eval_reward, iteration)
                writer.add_scalar("Eval/length", eval_length, iteration)

            if iteration % save_every == 0:
                torch.save(model.state_dict(), f"experiments/{exp_name}/model.pth")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        torch.save(model.state_dict(), f"experiments/{exp_name}/model.pth")

    writer.close()


if __name__ == "__main__":
    config_path = "src/rl/algorithms/actor_critic/config.yml"
    config = load_config(config_path)

    model = ActorCritic(**config["model"])
    optimizer = init_optimizer(config["optimizer"]["name"], model, **config["optimizer"]["kwargs"])

    os.makedirs(f"experiments/{config['exp_name']}", exist_ok=True)
    shutil.copy(config_path, f"experiments/{config['exp_name']}/config.yml")

    coefs_entropy = _get_entropy_schedule(
        int(config["num_iterations"]),
        int(config["coef_entropy_elbow"]),
        config["coef_entropy_max"],
        config["coef_entropy_min"],
    )

    print(f"Starting training: {config['exp_name']}")
    print(f"  num_envs={config['num_envs']}, rollout_length={config['rollout_length']}")
    print(f"  transitions/iter={config['num_envs'] * config['rollout_length']}")
    train(
        config["exp_name"],
        int(config["num_iterations"]),
        config["log_every"],
        config["save_every"],
        model,
        optimizer,
        config["rollout_length"],
        config["num_epochs"],
        config["minibatch_size"],
        config["clip_eps"],
        config["clip_value_loss"],
        config["gamma"],
        config["lam"],
        config["coef_value"],
        coefs_entropy,
        config["max_grad_norm"],
        config["num_envs"],
        torch.device("cpu"),
    )
