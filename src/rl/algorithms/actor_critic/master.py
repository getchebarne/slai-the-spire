"""
Master process for PPO training with parallel workers.

Coordinates multiple worker processes, collects trajectories, and updates the model.
"""

import os
import random
import shutil
from collections import deque
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Pipe
from torch.multiprocessing import Process
from torch.utils.tensorboard import SummaryWriter

from src.rl.action_space.masks import MaskBatch
from src.rl.action_space.masks import SELECTION_SIZES
from src.rl.action_space.masks import get_mask_batch
from src.rl.action_space.types import IS_DECISION_PRIMARY
from src.rl.action_space.types import NUM_PRIMARY_HEADS
from src.rl.action_space.types import PRIMARY_NUM_CHOICES
from src.rl.algorithms.actor_critic.worker import Command
from src.rl.algorithms.actor_critic.worker import WorkerData
from src.rl.algorithms.actor_critic.worker import worker
from src.rl.encoding.state import XGameState
from src.rl.encoding.state import encode_batch_view_game_state
from src.rl.models import ActorCritic
from src.rl.models.actor_critic import _build_entity_tensors
from src.rl.models.heads import compute_grouped_log_prob_and_entropy
from src.rl.utils import init_optimizer
from src.rl.utils import load_config


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class Transition:
    """Single transition in a trajectory. Stores pre-encoded state."""

    x_game_state: XGameState  # Pre-encoded state (batch=1)
    head_type_primary: int  # int(HeadTypePrimary) for fast indexing
    primary_mask: torch.Tensor | None  # (1, num_choices) for decision, None for direct
    selection_mask: torch.Tensor  # (1, max_entities)
    primary_index: int  # -1 for direct primaries
    primary_log_prob: torch.Tensor
    selection_index: int  # -1 if terminal
    selection_log_prob: torch.Tensor
    value: torch.Tensor
    reward: float


@dataclass
class Trajectory:
    """Sequence of transitions from one episode."""

    transitions: list[Transition] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.transitions)


@dataclass
class TrajectoryBatch:
    """Batched trajectory data for training."""

    x_game_states: list[XGameState]  # List of pre-encoded states (each batch=1)
    head_type_primaries: torch.Tensor  # (N,) int values of HeadTypePrimary
    primary_masks: list[torch.Tensor | None]  # each (1, num_choices) or None
    selection_masks: list[torch.Tensor]  # each (1, max_entities)
    primary_indices: torch.Tensor  # (N,)
    primary_log_probs: torch.Tensor  # (N,)
    selection_indices: torch.Tensor  # (N,)
    selection_log_probs: torch.Tensor  # (N,)
    values: torch.Tensor  # (N, 1)
    returns: torch.Tensor  # (N, 1)
    advantages: torch.Tensor  # (N, 1)

    def __len__(self) -> int:
        return len(self.x_game_states)


# =============================================================================
# XGameState Helpers
# =============================================================================


def _move_x_game_state(x: XGameState, device: torch.device) -> XGameState:
    """Move XGameState to a different device."""
    return XGameState(
        x_hand=x.x_hand.to(device),
        x_hand_mask_pad=x.x_hand_mask_pad.to(device),
        x_draw=x.x_draw.to(device),
        x_draw_mask_pad=x.x_draw_mask_pad.to(device),
        x_disc=x.x_disc.to(device),
        x_disc_mask_pad=x.x_disc_mask_pad.to(device),
        x_deck=x.x_deck.to(device),
        x_deck_mask_pad=x.x_deck_mask_pad.to(device),
        x_combat_reward=x.x_combat_reward.to(device),
        x_combat_reward_mask_pad=x.x_combat_reward_mask_pad.to(device),
        x_monsters=x.x_monsters.to(device),
        x_monsters_mask_pad=x.x_monsters_mask_pad.to(device),
        x_monster_health_block=x.x_monster_health_block.to(device),
        x_monster_modifiers=x.x_monster_modifiers.to(device),
        x_character=x.x_character.to(device),
        x_character_mask_pad=x.x_character_mask_pad.to(device),
        x_character_health_block=x.x_character_health_block.to(device),
        x_character_modifiers=x.x_character_modifiers.to(device),
        x_energy=x.x_energy.to(device),
        x_energy_mask_pad=x.x_energy_mask_pad.to(device),
        x_map=x.x_map.to(device),
        x_fsm=x.x_fsm.to(device),
    )


def _concat_x_game_states(x_game_states: list[XGameState]) -> XGameState:
    """Concatenate multiple XGameState objects along batch dimension."""
    return XGameState(
        x_hand=torch.cat([x.x_hand for x in x_game_states], dim=0),
        x_hand_mask_pad=torch.cat([x.x_hand_mask_pad for x in x_game_states], dim=0),
        x_draw=torch.cat([x.x_draw for x in x_game_states], dim=0),
        x_draw_mask_pad=torch.cat([x.x_draw_mask_pad for x in x_game_states], dim=0),
        x_disc=torch.cat([x.x_disc for x in x_game_states], dim=0),
        x_disc_mask_pad=torch.cat([x.x_disc_mask_pad for x in x_game_states], dim=0),
        x_deck=torch.cat([x.x_deck for x in x_game_states], dim=0),
        x_deck_mask_pad=torch.cat([x.x_deck_mask_pad for x in x_game_states], dim=0),
        x_combat_reward=torch.cat([x.x_combat_reward for x in x_game_states], dim=0),
        x_combat_reward_mask_pad=torch.cat(
            [x.x_combat_reward_mask_pad for x in x_game_states], dim=0
        ),
        x_monsters=torch.cat([x.x_monsters for x in x_game_states], dim=0),
        x_monsters_mask_pad=torch.cat([x.x_monsters_mask_pad for x in x_game_states], dim=0),
        x_monster_health_block=torch.cat([x.x_monster_health_block for x in x_game_states], dim=0),
        x_monster_modifiers=torch.cat([x.x_monster_modifiers for x in x_game_states], dim=0),
        x_character=torch.cat([x.x_character for x in x_game_states], dim=0),
        x_character_mask_pad=torch.cat([x.x_character_mask_pad for x in x_game_states], dim=0),
        x_character_health_block=torch.cat(
            [x.x_character_health_block for x in x_game_states], dim=0
        ),
        x_character_modifiers=torch.cat([x.x_character_modifiers for x in x_game_states], dim=0),
        x_energy=torch.cat([x.x_energy for x in x_game_states], dim=0),
        x_energy_mask_pad=torch.cat([x.x_energy_mask_pad for x in x_game_states], dim=0),
        x_map=torch.cat([x.x_map for x in x_game_states], dim=0),
        x_fsm=torch.cat([x.x_fsm for x in x_game_states], dim=0),
    )


def _slice_x_game_state(x_game_state: XGameState, idx: int) -> XGameState:
    """Slice a single sample from a batched XGameState."""
    return XGameState(
        x_hand=x_game_state.x_hand[idx : idx + 1],
        x_hand_mask_pad=x_game_state.x_hand_mask_pad[idx : idx + 1],
        x_draw=x_game_state.x_draw[idx : idx + 1],
        x_draw_mask_pad=x_game_state.x_draw_mask_pad[idx : idx + 1],
        x_disc=x_game_state.x_disc[idx : idx + 1],
        x_disc_mask_pad=x_game_state.x_disc_mask_pad[idx : idx + 1],
        x_deck=x_game_state.x_deck[idx : idx + 1],
        x_deck_mask_pad=x_game_state.x_deck_mask_pad[idx : idx + 1],
        x_combat_reward=x_game_state.x_combat_reward[idx : idx + 1],
        x_combat_reward_mask_pad=x_game_state.x_combat_reward_mask_pad[idx : idx + 1],
        x_monsters=x_game_state.x_monsters[idx : idx + 1],
        x_monsters_mask_pad=x_game_state.x_monsters_mask_pad[idx : idx + 1],
        x_monster_health_block=x_game_state.x_monster_health_block[idx : idx + 1],
        x_monster_modifiers=x_game_state.x_monster_modifiers[idx : idx + 1],
        x_character=x_game_state.x_character[idx : idx + 1],
        x_character_mask_pad=x_game_state.x_character_mask_pad[idx : idx + 1],
        x_character_health_block=x_game_state.x_character_health_block[idx : idx + 1],
        x_character_modifiers=x_game_state.x_character_modifiers[idx : idx + 1],
        x_energy=x_game_state.x_energy[idx : idx + 1],
        x_energy_mask_pad=x_game_state.x_energy_mask_pad[idx : idx + 1],
        x_map=x_game_state.x_map[idx : idx + 1],
        x_fsm=x_game_state.x_fsm[idx : idx + 1],
    )


# =============================================================================
# Mask Extraction Helpers
# =============================================================================


def _extract_per_sample_masks(
    mask_batch: MaskBatch,
    batch_idx: int,
    htp: int,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """
    Extract per-sample masks from a MaskBatch.

    Returns (primary_mask, selection_mask) each with batch dim 1.
    primary_mask is None for direct primaries.
    """
    # Find this sample's position within its group
    route = mask_batch.route[htp]
    local_idx = (route == batch_idx).nonzero(as_tuple=True)[0].item()

    primary_mask = None
    if IS_DECISION_PRIMARY[htp]:
        primary_mask = mask_batch.primary_masks[htp][local_idx : local_idx + 1]

    selection_mask = mask_batch.selection_masks[htp][local_idx : local_idx + 1]

    return primary_mask, selection_mask


def _build_mask_batch_from_samples(
    head_type_primaries: list[int],
    primary_masks: list[torch.Tensor | None],
    selection_masks: list[torch.Tensor],
    device: torch.device,
) -> MaskBatch:
    """Rebuild MaskBatch from per-sample data for PPO recomputation."""
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

    return MaskBatch(route=route, primary_masks=pm_list, selection_masks=sm_list)


# =============================================================================
# Episode Collection
# =============================================================================


def _run_episodes(
    model: ActorCritic, conn_parents: list[Connection], device: torch.device
) -> list[Trajectory]:
    """
    Run episodes in parallel across all workers.

    Returns a list of trajectories, one per worker.
    """
    num_envs = len(conn_parents)

    # Reset all environments
    for conn in conn_parents:
        conn.send((Command.RESET, None))

    # Get initial states
    worker_datas: list[WorkerData] = [conn.recv() for conn in conn_parents]

    # Initialize trajectories
    trajectories = [Trajectory() for _ in range(num_envs)]

    # Track which envs are still running
    running_envs = list(range(num_envs))

    model.eval()
    with torch.no_grad():
        while running_envs:
            # Collect non-terminal states
            active_envs = []
            view_game_states = []

            for env_idx in running_envs:
                data = worker_datas[env_idx]
                if not data.game_over:
                    active_envs.append(env_idx)
                    view_game_states.append(data.view_game_state)

            running_envs = active_envs
            if not running_envs:
                break

            # Encode all states at once
            x_game_state = encode_batch_view_game_state(view_game_states, device)

            # Build MaskBatch (routes + masks)
            mask_batch = get_mask_batch(view_game_states, device)

            # Forward pass
            output = model(x_game_state, mask_batch, sample=True)

            # Send actions to workers
            for i, env_idx in enumerate(active_envs):
                action = output.get_action(i)
                conn_parents[env_idx].send((Command.STEP, action))

            # Receive new states
            new_worker_datas = {}
            for env_idx in running_envs:
                new_worker_datas[env_idx] = conn_parents[env_idx].recv()

            # Store transitions
            for i, env_idx in enumerate(running_envs):
                reward = new_worker_datas[env_idx].reward

                htp = output.head_type_primaries[i].item()
                primary_mask, selection_mask = _extract_per_sample_masks(mask_batch, i, htp)

                transition = Transition(
                    x_game_state=_slice_x_game_state(x_game_state, i),
                    head_type_primary=htp,
                    primary_mask=primary_mask,
                    selection_mask=selection_mask,
                    primary_index=output.primary_indices[i].item(),
                    primary_log_prob=output.primary_log_probs[i],
                    selection_index=output.selection_indices[i].item(),
                    selection_log_prob=output.selection_log_probs[i],
                    value=output.values[i],
                    reward=reward,
                )
                trajectories[env_idx].transitions.append(transition)

            # Update worker data
            for env_idx in running_envs:
                worker_datas[env_idx] = new_worker_datas[env_idx]

    model.train()
    return trajectories


def _run_eval_episode(
    model: ActorCritic,
    conn: Connection,
    device: torch.device,
) -> tuple[float, int]:
    """
    Run a single greedy evaluation episode (sample=False).

    Returns (total_reward, episode_length).
    """
    conn.send((Command.RESET, None))
    data: WorkerData = conn.recv()

    total_reward = 0.0
    episode_length = 0

    model.eval()
    with torch.no_grad():
        while not data.game_over:
            x_game_state = encode_batch_view_game_state([data.view_game_state], device)
            mask_batch = get_mask_batch([data.view_game_state], device)

            output = model(x_game_state, mask_batch, sample=False)

            action = output.get_action(0)
            conn.send((Command.STEP, action))
            data = conn.recv()

            total_reward += data.reward
            episode_length += 1

    model.train()
    return total_reward, episode_length


# =============================================================================
# GAE and Batch Creation
# =============================================================================


def _compute_gae(
    rewards: list[float],
    values: list[torch.Tensor],
    gamma: float,
    lam: float,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    """Compute returns and GAE advantages."""
    returns = deque()
    advantages = deque()

    values_with_terminal = values + [torch.tensor([[0.0]], device=device)]

    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = (
            rewards[t]
            + gamma * values_with_terminal[t + 1].item()
            - values_with_terminal[t].item()
        )
        gae = delta + gamma * lam * gae
        returns.appendleft(gae + values_with_terminal[t].item())
        advantages.appendleft(gae)

    return list(returns), list(advantages)


def _create_batch(
    trajectories: list[Trajectory],
    gamma: float,
    lam: float,
    device: torch.device,
) -> TrajectoryBatch:
    """Create a training batch from trajectories."""
    all_x_game_states = []
    all_head_type_primaries = []
    all_primary_masks = []
    all_selection_masks = []
    all_primary_indices = []
    all_primary_log_probs = []
    all_selection_indices = []
    all_selection_log_probs = []
    all_values = []
    all_returns = []
    all_advantages = []

    for trajectory in trajectories:
        if len(trajectory) == 0:
            continue

        rewards = [t.reward for t in trajectory.transitions]
        values = [t.value for t in trajectory.transitions]
        returns, advantages = _compute_gae(rewards, values, gamma, lam, device)

        for i, trans in enumerate(trajectory.transitions):
            all_x_game_states.append(trans.x_game_state)
            all_head_type_primaries.append(trans.head_type_primary)
            all_primary_masks.append(trans.primary_mask)
            all_selection_masks.append(trans.selection_mask)
            all_primary_indices.append(trans.primary_index)
            all_primary_log_probs.append(trans.primary_log_prob.item())
            all_selection_indices.append(trans.selection_index)
            all_selection_log_probs.append(trans.selection_log_prob.item())
            all_values.append(trans.value)
            all_returns.append(returns[i])
            all_advantages.append(advantages[i])

    batch = TrajectoryBatch(
        x_game_states=all_x_game_states,
        head_type_primaries=torch.tensor(all_head_type_primaries, dtype=torch.long, device=device),
        primary_masks=all_primary_masks,
        selection_masks=all_selection_masks,
        primary_indices=torch.tensor(all_primary_indices, dtype=torch.long, device=device),
        primary_log_probs=torch.tensor(all_primary_log_probs, dtype=torch.float32, device=device),
        selection_indices=torch.tensor(all_selection_indices, dtype=torch.long, device=device),
        selection_log_probs=torch.tensor(
            all_selection_log_probs, dtype=torch.float32, device=device
        ),
        values=torch.cat(all_values, dim=0).detach(),
        returns=torch.tensor(all_returns, dtype=torch.float32, device=device).view(-1, 1),
        advantages=torch.tensor(all_advantages, dtype=torch.float32, device=device).view(-1, 1),
    )

    # Normalize advantages
    batch.advantages = (batch.advantages - torch.mean(batch.advantages)) / (
        torch.std(batch.advantages) + 1e-8
    )

    return batch


# =============================================================================
# PPO Update
# =============================================================================


def _minibatch_indices(total: int, minibatch_size: int) -> Iterator[list[int]]:
    """Generate shuffled minibatch indices."""
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
    primary_indices: torch.Tensor,
    selection_indices: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Recompute log probs and entropy for a minibatch with current policy.

    Uses direct tensor field access (no full CoreOutput slicing).
    Returns (log_probs, entropies, values) - all shape (N,) or (N, 1)
    """
    B = len(x_game_states)

    # Concatenate pre-encoded states and move to device
    x_game_state = _concat_x_game_states(x_game_states)
    x_game_state = _move_x_game_state(x_game_state, device)

    # Rebuild MaskBatch for this minibatch
    mask_batch = _build_mask_batch_from_samples(
        head_type_primaries, primary_masks, selection_masks, device
    )

    primary_indices = primary_indices.to(device)
    selection_indices = selection_indices.to(device)

    # Core encoder (all samples)
    core_out = model.core(x_game_state)

    # Value head (all samples)
    values = model.head_value(core_out.x_global)

    # Pre-extract entity tensors (just references, no copy)
    entity_tensors = _build_entity_tensors(core_out)

    # Initialize log probs and entropies
    total_log_probs = torch.zeros(B, device=device)
    total_entropies = torch.zeros(B, device=device)

    # Process each primary group
    for htp in range(NUM_PRIMARY_HEADS):
        idx = mask_batch.route[htp]
        if len(idx) == 0:
            continue

        x_global_group = core_out.x_global[idx]

        if IS_DECISION_PRIMARY[htp]:
            # --- Recompute primary (binary) decision ---
            primary_mask = mask_batch.primary_masks[htp]
            decision_head = model._decision_heads[htp]
            out = decision_head(x_global_group, primary_mask, sample=False)

            primary_dist = torch.distributions.Categorical(logits=out.logits)
            group_primary_idx = primary_indices[idx]
            total_log_probs[idx] += primary_dist.log_prob(group_primary_idx)
            total_entropies[idx] += primary_dist.entropy()

            # --- Recompute secondary selection (for samples that chose "select") ---
            needs_secondary = group_primary_idx == 1
            if torch.any(needs_secondary):
                sec_local = torch.nonzero(needs_secondary, as_tuple=True)[0]
                sec_batch = idx[sec_local]

                sec_x_global = x_global_group[sec_local]
                sec_entities = entity_tensors[htp][sec_batch]
                sec_mask = mask_batch.selection_masks[htp][sec_local]
                sec_indices = selection_indices[sec_batch]

                sel_head = model._selection_heads[htp]
                out = sel_head(sec_entities, sec_x_global, sec_mask, sample=False)
                sec_log_probs, sec_entropy = compute_grouped_log_prob_and_entropy(
                    out.logits, sec_indices
                )

                total_log_probs[sec_batch] += sec_log_probs
                total_entropies[sec_batch] += sec_entropy

        else:
            # --- Direct primary: recompute selection ---
            entities_group = entity_tensors[htp][idx]
            sel_mask = mask_batch.selection_masks[htp]
            sel_indices = selection_indices[idx]

            sel_head = model._selection_heads[htp]
            out = sel_head(entities_group, x_global_group, sel_mask, sample=False)
            sel_log_probs, sel_entropy = compute_grouped_log_prob_and_entropy(
                out.logits, sel_indices
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
    """PPO update step."""
    total_loss_policy = 0.0
    total_loss_value = 0.0
    total_loss_entropy = 0.0
    num_updates = 0

    for _ in range(num_epochs):
        for mb_idxs in _minibatch_indices(len(batch), minibatch_size):
            # Gather minibatch data
            mb_x_states = [batch.x_game_states[i] for i in mb_idxs]
            mb_htps = [batch.head_type_primaries[i].item() for i in mb_idxs]
            mb_primary_masks = [batch.primary_masks[i] for i in mb_idxs]
            mb_selection_masks = [batch.selection_masks[i] for i in mb_idxs]
            mb_primary_indices = batch.primary_indices[mb_idxs]
            mb_selection_indices = batch.selection_indices[mb_idxs]

            # Recompute with current policy
            log_probs_new, entropies, values_new = _recompute_log_probs_batch(
                model,
                mb_x_states,
                mb_htps,
                mb_primary_masks,
                mb_selection_masks,
                mb_primary_indices,
                mb_selection_indices,
                device,
            )

            # Old log probs
            log_probs_old = batch.primary_log_probs[mb_idxs].to(
                device
            ) + batch.selection_log_probs[mb_idxs].to(device)

            # Advantages and returns
            advantages = torch.squeeze(batch.advantages[mb_idxs]).to(device)
            returns = batch.returns[mb_idxs].to(device)
            values_old = batch.values[mb_idxs].to(device)

            # Policy loss (PPO clipped)
            ratio = torch.exp(log_probs_new - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            loss_policy = -torch.mean(torch.min(surr1, surr2))

            # Value loss
            if clip_value_loss:
                values_clipped = values_old + torch.clamp(
                    values_new - values_old, -clip_eps, clip_eps
                )
                loss_value_unclipped = torch.pow(values_new - returns, 2)
                loss_value_clipped = torch.pow(values_clipped - returns, 2)
                loss_value = 0.5 * torch.mean(torch.max(loss_value_unclipped, loss_value_clipped))
            else:
                loss_value = F.mse_loss(values_new, returns)

            # Entropy loss
            loss_entropy = -torch.mean(entropies)

            # Total loss
            loss = loss_policy + coef_value * loss_value + coef_entropy * loss_entropy

            # Backward
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
    num_episodes: int,
    elbow: int,
    max_coef: float,
    min_coef: float,
) -> list[float]:
    """Linear decay of entropy coefficient."""
    coefs = []
    slope = (min_coef - max_coef) / elbow

    for ep in range(num_episodes):
        if ep <= elbow:
            coefs.append(slope * ep + max_coef)
        else:
            coefs.append(min_coef)

    return coefs


def train(
    exp_name: str,
    num_episodes: int,
    log_every: int,
    save_every: int,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
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
    """Main training loop."""
    writer = SummaryWriter(f"experiments/{exp_name}")
    model.to(device)

    # Start worker processes
    conn_parents, conn_children = zip(*[Pipe() for _ in range(num_envs)])
    processes = [Process(target=worker, args=(conn,)) for conn in conn_children]
    for p in processes:
        p.start()

    try:
        for episode in range(num_episodes):
            coef_entropy = coefs_entropy[episode]

            # Collect trajectories
            trajectories = _run_episodes(model, list(conn_parents), device)

            # Create batch
            batch = _create_batch(trajectories, gamma, lam, device)

            if len(batch) == 0:
                print(f"Episode {episode}: Empty batch, skipping")
                continue

            # PPO update
            loss_policy, loss_value, loss_entropy = _update_ppo(
                model,
                batch,
                optimizer,
                num_epochs,
                minibatch_size,
                clip_eps,
                clip_value_loss,
                coef_value,
                coef_entropy,
                max_grad_norm,
                device,
            )

            # Logging
            if episode % log_every == 0:
                print(f"Episode {episode}: policy={loss_policy:.4f}, value={loss_value:.4f}")
                writer.add_scalar("Loss/policy", loss_policy, episode)
                writer.add_scalar("Loss/value", loss_value, episode)
                writer.add_scalar("Loss/entropy", loss_entropy, episode)
                writer.add_scalar("Entropy/coef", coef_entropy, episode)

                total_reward = sum(
                    sum(t.reward for t in traj.transitions) for traj in trajectories
                )
                avg_length = sum(len(traj) for traj in trajectories) / max(len(trajectories), 1)
                writer.add_scalar("Trajectory/total_reward", total_reward, episode)
                writer.add_scalar("Trajectory/avg_length", avg_length, episode)

                # Greedy evaluation episode
                eval_reward, eval_length = _run_eval_episode(model, conn_parents[0], device)
                print(f"  eval: reward={eval_reward:.4f}, length={eval_length}")
                writer.add_scalar("Eval/reward", eval_reward, episode)
                writer.add_scalar("Eval/length", eval_length, episode)

            # Save
            if episode % save_every == 0:
                torch.save(model.state_dict(), f"experiments/{exp_name}/model.pth")

    finally:
        for conn in conn_parents:
            conn.send((Command.CLOSE, None))
        for p in processes:
            p.join()

    writer.close()


if __name__ == "__main__":
    config_path = "src/rl/algorithms/actor_critic/config.yml"
    config = load_config(config_path)

    # Model
    model = ActorCritic(**config["model"])

    # Optimizer
    optimizer = init_optimizer(config["optimizer"]["name"], model, **config["optimizer"]["kwargs"])

    # Create experiment directory
    os.makedirs(f"experiments/{config['exp_name']}", exist_ok=True)
    shutil.copy(config_path, f"experiments/{config['exp_name']}/config.yml")

    # Entropy schedule
    coefs_entropy = _get_entropy_schedule(
        int(config["num_episodes"]),
        int(config["coef_entropy_elbow"]),
        config["coef_entropy_max"],
        config["coef_entropy_min"],
    )

    # Train
    print(f"Starting training: {config['exp_name']}")
    train(
        config["exp_name"],
        int(config["num_episodes"]),
        config["log_every"],
        config["save_every"],
        model,
        optimizer,
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
