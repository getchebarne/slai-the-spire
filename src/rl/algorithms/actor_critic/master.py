"""
PPO training with fixed-length rollouts and in-process environments.

No multiprocessing -- all game environments run in the main process.
Each iteration collects a fixed number of steps from all environments,
auto-resetting games that end. GAE handles episode boundaries correctly.
"""

import os
import random
import shutil
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.game.action import Action
from src.game.core.fsm import FSM
from src.game.create import create_game_state
from src.game.main import initialize_game_state
from src.game.main import step as game_step
from src.game.view.state import ViewGameState
from src.game.view.state import get_view_game_state
from src.rl.action_space.masks import MaskBatch
from src.rl.action_space.masks import SELECTION_SIZES
from src.rl.action_space.masks import get_mask_batch
from src.rl.action_space.types import HeadTypePrimary
from src.rl.action_space.types import IS_DECISION_PRIMARY
from src.rl.action_space.types import NUM_PRIMARY_HEADS
from src.rl.action_space.types import PRIMARY_NUM_CHOICES
from src.rl.action_space.types import to_action
from src.rl.constants import ASCENSION_LEVEL
from src.rl.encoding.state import XGameState
from src.rl.encoding.state import encode_batch_view_game_state
from src.rl.models import ActorCritic
from src.rl.models.actor_critic import _build_entity_tensors
from src.rl.models.heads import compute_grouped_log_prob_and_entropy
from src.rl.reward import compute_reward
from src.rl.utils import init_optimizer
from src.rl.utils import load_config


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class Transition:
    """Single transition with done flag for episode boundaries."""

    x_game_state: XGameState  # Pre-encoded state (batch=1)
    head_type_primary: int  # int(HeadTypePrimary)
    primary_mask: torch.Tensor | None  # (1, num_choices) or None
    selection_mask: torch.Tensor  # (1, max_entities)
    primary_index: int  # -1 for direct primaries
    primary_log_prob: torch.Tensor
    selection_index: int  # -1 if terminal
    selection_log_prob: torch.Tensor
    value: torch.Tensor
    reward: float
    done: bool  # True if episode ended at this step


@dataclass
class TrajectoryBatch:
    """Batched data for PPO training."""

    x_game_states: list[XGameState]
    head_type_primaries: torch.Tensor  # (N,)
    primary_masks: list[torch.Tensor | None]
    selection_masks: list[torch.Tensor]
    primary_indices: torch.Tensor  # (N,)
    primary_log_probs: torch.Tensor  # (N,)
    selection_indices: torch.Tensor  # (N,)
    selection_log_probs: torch.Tensor  # (N,)
    values: torch.Tensor  # (N, 1)
    returns: torch.Tensor  # (N, 1)
    advantages: torch.Tensor  # (N, 1)

    def __len__(self) -> int:
        return len(self.x_game_states)


@dataclass
class EpisodeStats:
    """Stats from a completed episode (for logging)."""

    total_reward: float
    length: int


# =============================================================================
# In-Process Environment Manager
# =============================================================================


class EnvironmentManager:
    """
    Manages N game environments in-process (no IPC).

    Environments persist across rollouts. Games that end are auto-reset.
    """

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self._game_states = []
        self._view_states: list[ViewGameState] = []

        for _ in range(num_envs):
            gs, vgs = self._make_env()
            self._game_states.append(gs)
            self._view_states.append(vgs)

    @staticmethod
    def _make_env():
        gs = create_game_state(ASCENSION_LEVEL)
        initialize_game_state(gs)
        vgs = get_view_game_state(gs)
        return gs, vgs

    def get_view_states(self) -> list[ViewGameState]:
        return self._view_states

    def step(self, env_idx: int, action: Action) -> tuple[float, bool]:
        """
        Step environment, auto-reset on game over.

        Returns (reward, done).
        """
        gs = self._game_states[env_idx]
        vgs_prev = self._view_states[env_idx]

        game_step(gs, action, fast_mode=True)
        vgs_next = get_view_game_state(gs)
        done = gs.fsm == FSM.GAME_OVER

        reward = compute_reward(vgs_prev, vgs_next, done)

        if done:
            # Auto-reset
            gs_new, vgs_new = self._make_env()
            self._game_states[env_idx] = gs_new
            self._view_states[env_idx] = vgs_new
        else:
            self._view_states[env_idx] = vgs_next

        return reward, done


# =============================================================================
# XGameState Helpers
# =============================================================================


def _move_x_game_state(x: XGameState, device: torch.device) -> XGameState:
    """Move XGameState to a different device."""
    return XGameState(
        x_hand=x.x_hand.to(device),
        x_hand_mask_pad=x.x_hand_mask_pad.to(device),
        x_active_card_mask=x.x_active_card_mask.to(device),
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
        x_active_card_mask=torch.cat([x.x_active_card_mask for x in x_game_states], dim=0),
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
        x_monster_health_block=torch.cat(
            [x.x_monster_health_block for x in x_game_states], dim=0
        ),
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
        x_active_card_mask=x_game_state.x_active_card_mask[idx : idx + 1],
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
# Mask Helpers
# =============================================================================


def _build_route_map(mask_batch: MaskBatch) -> dict[tuple[int, int], int]:
    """
    Build a reverse-index map from (htp, batch_idx) -> local_idx.

    One bulk .cpu().tolist() per active head type instead of
    nonzero().item() per env (eliminates ~2 syncs per env per step).
    """
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
    """Extract per-sample masks from a MaskBatch using pre-computed route map."""
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
# Rollout Collection
# =============================================================================


def _collect_rollout(
    model: ActorCritic,
    env_mgr: EnvironmentManager,
    rollout_length: int,
    device: torch.device,
) -> tuple[list[list[Transition]], list[torch.Tensor], list[EpisodeStats]]:
    """
    Collect a fixed-length rollout from all environments.

    Each env takes exactly `rollout_length` steps. Games that end are
    auto-reset and collection continues. Environments persist across calls.

    Returns:
        transitions: [num_envs][rollout_length] transitions
        bootstrap_values: [num_envs] V(s') for truncated rollouts (GAE)
        completed_episodes: stats for episodes that finished during this rollout
    """
    num_envs = env_mgr.num_envs
    transitions: list[list[Transition]] = [[] for _ in range(num_envs)]
    completed_episodes: list[EpisodeStats] = []

    # Track running episode stats for logging
    ep_rewards = [0.0] * num_envs
    ep_lengths = [0] * num_envs

    model.eval()
    with torch.no_grad():
        for _ in range(rollout_length):
            view_states = env_mgr.get_view_states()

            # Batch encode + forward
            x_game_state = encode_batch_view_game_state(view_states, device)
            mask_batch = get_mask_batch(view_states, device)
            output = model(x_game_state, mask_batch, sample=True)

            # Bulk-extract discrete outputs: 3 syncs instead of 8 * num_envs
            htps = output.head_type_primaries.cpu().tolist()
            pis = output.primary_indices.cpu().tolist()
            sis = output.selection_indices.cpu().tolist()

            # Pre-compute route reverse map (eliminates nonzero().item() per env)
            route_map = _build_route_map(mask_batch)

            # Step each environment
            for i in range(num_envs):
                htp = htps[i]
                action = to_action(HeadTypePrimary(htp), pis[i], sis[i])
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
                        value=output.values[i],
                        reward=reward,
                        done=done,
                    )
                )

                # Track episode stats
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
    """
    Run a single greedy evaluation episode (no workers needed).

    Returns (total_reward, episode_length).
    """
    gs = create_game_state(ASCENSION_LEVEL)
    initialize_game_state(gs)

    total_reward = 0.0
    length = 0

    model.eval()
    with torch.no_grad():
        while gs.fsm != FSM.GAME_OVER:
            vgs = get_view_game_state(gs)
            x = encode_batch_view_game_state([vgs], device)
            mb = get_mask_batch([vgs], device)
            output = model(x, mb, sample=False)
            action = output.get_action(0)

            vgs_prev = vgs
            game_step(gs, action, fast_mode=True)
            vgs_next = get_view_game_state(gs)
            done = gs.fsm == FSM.GAME_OVER
            reward = compute_reward(vgs_prev, vgs_next, done)

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
    """
    Compute returns and GAE advantages with episode boundary handling.

    At done=True boundaries: bootstrap with 0 (episode ended).
    At rollout end (truncation): bootstrap with bootstrap_value.
    """
    T = len(rewards)
    advantages = [0.0] * T
    returns = [0.0] * T

    # Bulk-transfer values to CPU: 1 sync instead of ~3*T syncs
    values_cpu = torch.stack(values).squeeze(-1).cpu().tolist()
    bootstrap_cpu = bootstrap_value.cpu().item()

    gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = bootstrap_cpu
        else:
            next_value = values_cpu[t + 1]

        # If episode ended at step t, don't bootstrap from next state
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
    """Create a training batch from rollout transitions with GAE."""
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

    x_game_state = _concat_x_game_states(x_game_states)
    x_game_state = _move_x_game_state(x_game_state, device)

    mask_batch = _build_mask_batch_from_samples(
        head_type_primaries, primary_masks, selection_masks, device
    )

    primary_indices = primary_indices.to(device)
    selection_indices = selection_indices.to(device)

    core_out = model.core(x_game_state)
    values = model.head_value(core_out.x_global)
    entity_tensors = _build_entity_tensors(core_out)

    total_log_probs = torch.zeros(B, device=device)
    total_entropies = torch.zeros(B, device=device)

    for htp in range(NUM_PRIMARY_HEADS):
        idx = mask_batch.route[htp]
        if len(idx) == 0:
            continue

        x_global_group = core_out.x_global[idx]

        if IS_DECISION_PRIMARY[htp]:
            primary_mask = mask_batch.primary_masks[htp]
            decision_head = model._decision_heads[htp]
            out = decision_head(x_global_group, primary_mask, sample=False)

            primary_dist = torch.distributions.Categorical(logits=out.logits)
            group_primary_idx = primary_indices[idx]
            total_log_probs[idx] += primary_dist.log_prob(group_primary_idx)
            total_entropies[idx] += primary_dist.entropy()

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
            entities_group = entity_tensors[htp][idx]
            sel_mask = mask_batch.selection_masks[htp]
            sel_indices = selection_indices[idx]

            sel_head = model._selection_heads[htp]

            # Monster select gets active card embedding for card-dependent targeting
            if htp == int(HeadTypePrimary.COMBAT_MONSTER_SELECT):
                out = sel_head(
                    entities_group, x_global_group, sel_mask, sample=False,
                    x_active_card=core_out.x_active_card[idx],
                )
            else:
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
            mb_x_states = [batch.x_game_states[i] for i in mb_idxs]
            mb_htps = batch.head_type_primaries[mb_idxs].cpu().tolist()
            mb_primary_masks = [batch.primary_masks[i] for i in mb_idxs]
            mb_selection_masks = [batch.selection_masks[i] for i in mb_idxs]
            mb_primary_indices = batch.primary_indices[mb_idxs]
            mb_selection_indices = batch.selection_indices[mb_idxs]

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

            log_probs_old = batch.primary_log_probs[mb_idxs].to(
                device
            ) + batch.selection_log_probs[mb_idxs].to(device)

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
    """Linear decay of entropy coefficient."""
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
    """Main training loop with fixed-length rollouts."""
    writer = SummaryWriter(f"experiments/{exp_name}")
    model.to(device)

    # Create in-process environment manager
    env_mgr = EnvironmentManager(num_envs)

    total_steps = 0

    try:
        for iteration in range(num_iterations):
            coef_entropy = coefs_entropy[iteration]

            # Collect fixed-length rollout
            transitions, bootstrap_values, completed_episodes = _collect_rollout(
                model, env_mgr, rollout_length, device
            )

            num_transitions = rollout_length * num_envs
            total_steps += num_transitions

            # Create batch with GAE
            batch = _create_batch(transitions, bootstrap_values, gamma, lam, device)

            if len(batch) == 0:
                print(f"Iteration {iteration}: Empty batch, skipping")
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
                    avg_reward = sum(e.total_reward for e in completed_episodes) / len(
                        completed_episodes
                    )
                    avg_length = sum(e.length for e in completed_episodes) / len(
                        completed_episodes
                    )
                    writer.add_scalar("Episode/avg_reward", avg_reward, iteration)
                    writer.add_scalar("Episode/avg_length", avg_length, iteration)
                    writer.add_scalar(
                        "Episode/completed_count", len(completed_episodes), iteration
                    )

                # Greedy evaluation episode
                eval_reward, eval_length = _run_eval_episode(model, device)
                print(f"  eval: reward={eval_reward:.4f}, length={eval_length}")
                writer.add_scalar("Eval/reward", eval_reward, iteration)
                writer.add_scalar("Eval/length", eval_length, iteration)

            # Save
            if iteration % save_every == 0:
                torch.save(model.state_dict(), f"experiments/{exp_name}/model.pth")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        torch.save(model.state_dict(), f"experiments/{exp_name}/model.pth")

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
        int(config["num_iterations"]),
        int(config["coef_entropy_elbow"]),
        config["coef_entropy_max"],
        config["coef_entropy_min"],
    )

    # Train
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
