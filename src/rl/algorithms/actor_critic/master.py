"""PPO training with fixed-length rollouts and in-process environments.

Envs run in-process with `fast_mode=True` (the engine auto-advances trivial
single-legal-action states, so the trainer only sees real choice points).
`env.step` returns a 2-tuple `(obs, terminated)`; reward is computed trainer-side.

Action masks are derived from each env's `get_legal_actions()` (the authoritative
legal set) and stored per transition so PPO recompute needs no live envs.
"""

import os
import random
import shutil
from dataclasses import dataclass
from typing import Iterator

import slai
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.rl.action_space.masks import SEL_GROUP_PILE
from src.rl.action_space.masks import SEL_POOL_SIZE
from src.rl.action_space.masks import MaskBatch
from src.rl.action_space.masks import get_mask_batch
from src.rl.action_space.types import NUM_PRIMARY_HEADS
from src.rl.action_space.types import NUM_SEL_KEYS
from src.rl.action_space.types import PRIMARY_NUM_CHOICES
from src.rl.action_space.types import HeadTypePrimary
from src.rl.constants import ASCENSION_LEVEL
from src.rl.constants import FAST_MODE
from src.rl.constants import MAX_MONSTERS
from src.rl.constants import MAX_SIZE_HAND
from src.rl.encoding.state import TensorGameState
from src.rl.encoding.state import encode_batch_game_state
from src.rl.models import ActorCritic
from src.rl.reward import compute_reward
from src.rl.utils import init_optimizer
from src.rl.utils import load_config


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class Transition:
    x_game_state: TensorGameState  # batch=1
    head_type_primary: int
    option_index: int
    sel_key: int
    selection_index: int
    target_index: int
    retain_indices: torch.Tensor  # (1, MAX_SIZE_HAND)
    # Stored masks for PPO recompute
    option_mask: torch.Tensor | None  # (1, K_htp)
    sel_mask: torch.Tensor | None  # (1, pool)
    sel_group_ids: torch.Tensor | None  # (1, pool)
    monster_alive_mask: torch.Tensor  # (1, MAX_MONSTERS)
    multipick_mask: torch.Tensor  # (1, MAX_SIZE_HAND)
    multipick_group_ids: torch.Tensor  # (1, MAX_SIZE_HAND)
    pick_num: int
    log_prob_old: torch.Tensor  # scalar
    value: torch.Tensor
    reward: float
    done: bool


@dataclass
class Rec:
    """Recorded action indices for ActorCritic.evaluate_actions."""

    head_type_primaries: torch.Tensor
    option_indices: torch.Tensor
    selection_indices: torch.Tensor
    target_indices: torch.Tensor
    retain_indices: torch.Tensor


@dataclass
class EpisodeStats:
    total_reward: float
    length: int


# =============================================================================
# Environment manager
# =============================================================================


class EnvironmentManager:
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
        env = slai.GameEnv(ascension=ASCENSION_LEVEL, fast_mode=FAST_MODE)
        obs = env.reset(seed=random.randint(0, 2**31 - 1))
        return env, obs

    def get_view_states(self) -> list[slai.GameState]:
        return self._obs

    def get_legal_actions(self) -> list[list]:
        return [env.get_legal_actions() for env in self._envs]

    def step(self, env_idx: int, action) -> tuple[float, bool]:
        prev = self._obs[env_idx]
        nxt, terminated = self._envs[env_idx].step(action)
        reward = compute_reward(prev, nxt, terminated)
        if terminated:
            env, nxt = self._make_env()
            self._envs[env_idx] = env
        self._obs[env_idx] = nxt
        return reward, terminated


# =============================================================================
# TensorGameState helpers (generic over the dataclass fields)
# =============================================================================


def _move_x_game_state(x: TensorGameState, device: torch.device) -> TensorGameState:
    return x.to(device)


def _concat_x_game_states(xs: list[TensorGameState]) -> TensorGameState:
    return torch.cat(xs, dim=0)


def _slice_x_game_state(x: TensorGameState, idx: int) -> TensorGameState:
    return x[idx : idx + 1]


# =============================================================================
# Mask (de)composition for recompute
# =============================================================================


def _build_mask_batch_from_samples(
    transitions: list[Transition], device: torch.device
) -> MaskBatch:
    """Rebuild a full-batch MaskBatch from stored per-sample data. Only the
    recorded path's masks are populated (recompute evaluates recorded actions)."""
    N = len(transitions)
    htps = [t.head_type_primary for t in transitions]

    route_lists: list[list[int]] = [[] for _ in range(NUM_PRIMARY_HEADS)]
    for i, htp in enumerate(htps):
        route_lists[htp].append(i)
    route = [
        torch.tensor(route_lists[h], dtype=torch.long, device=device)
        for h in range(NUM_PRIMARY_HEADS)
    ]

    # Option masks per screen htp (stacked in route order)
    option_masks = []
    for htp in range(NUM_PRIMARY_HEADS):
        k = PRIMARY_NUM_CHOICES[htp]
        idxs = route_lists[htp]
        if k == 0 or not idxs:
            option_masks.append(torch.zeros(len(idxs), k, dtype=torch.bool, device=device))
        else:
            option_masks.append(
                torch.cat([transitions[i].option_mask for i in idxs], dim=0).to(device)
            )

    # Selection masks + group ids per SelKey (full batch; only recorded rows filled)
    sel_masks = [
        torch.zeros(N, SEL_POOL_SIZE[k], dtype=torch.bool, device=device)
        for k in range(NUM_SEL_KEYS)
    ]
    sel_group_ids = [
        torch.full((N, SEL_POOL_SIZE[k]), -1, dtype=torch.long, device=device)
        if SEL_GROUP_PILE[k] is not None
        else None
        for k in range(NUM_SEL_KEYS)
    ]
    for i, t in enumerate(transitions):
        if t.sel_key >= 0 and t.sel_mask is not None:
            sel_masks[t.sel_key][i] = t.sel_mask[0].to(device)
            if sel_group_ids[t.sel_key] is not None and t.sel_group_ids is not None:
                sel_group_ids[t.sel_key][i] = t.sel_group_ids[0].to(device)

    monster_alive_mask = torch.cat([t.monster_alive_mask for t in transitions], dim=0).to(device)
    multipick_mask = torch.cat([t.multipick_mask for t in transitions], dim=0).to(device)
    multipick_group_ids = torch.cat([t.multipick_group_ids for t in transitions], dim=0).to(device)
    pick_nums = torch.tensor([t.pick_num for t in transitions], dtype=torch.long, device=device)

    return MaskBatch(
        route=route,
        option_masks=option_masks,
        sel_masks=sel_masks,
        sel_group_ids=sel_group_ids,
        monster_alive_mask=monster_alive_mask,
        target_required_hand=torch.zeros(N, MAX_SIZE_HAND, dtype=torch.bool, device=device),
        target_required_potion=torch.zeros(N, 0, dtype=torch.bool, device=device),
        multipick_mask=multipick_mask,
        multipick_group_ids=multipick_group_ids,
        pick_nums=pick_nums,
    )


def _rec_from_samples(transitions: list[Transition], device: torch.device) -> Rec:
    return Rec(
        head_type_primaries=torch.tensor(
            [t.head_type_primary for t in transitions], dtype=torch.long, device=device
        ),
        option_indices=torch.tensor(
            [t.option_index for t in transitions], dtype=torch.long, device=device
        ),
        selection_indices=torch.tensor(
            [t.selection_index for t in transitions], dtype=torch.long, device=device
        ),
        target_indices=torch.tensor(
            [t.target_index for t in transitions], dtype=torch.long, device=device
        ),
        retain_indices=torch.cat([t.retain_indices for t in transitions], dim=0).to(device),
    )


# =============================================================================
# Rollout collection
# =============================================================================


def _build_route_map(route: list[torch.Tensor]) -> dict[tuple[int, int], int]:
    route_map: dict[tuple[int, int], int] = {}
    for htp in range(NUM_PRIMARY_HEADS):
        for local_idx, batch_idx in enumerate(route[htp].cpu().tolist()):
            route_map[(htp, batch_idx)] = local_idx
    return route_map


def _collect_rollout(
    model: ActorCritic,
    env_mgr: EnvironmentManager,
    rollout_length: int,
    device: torch.device,
) -> tuple[list[list[Transition]], list[torch.Tensor], list[EpisodeStats]]:
    num_envs = env_mgr.num_envs
    transitions: list[list[Transition]] = [[] for _ in range(num_envs)]
    completed: list[EpisodeStats] = []
    ep_rewards = [0.0] * num_envs
    ep_lengths = [0] * num_envs

    model.eval()
    with torch.no_grad():
        for _ in range(rollout_length):
            view_states = env_mgr.get_view_states()
            legal_batch = env_mgr.get_legal_actions()
            x_game_state = encode_batch_game_state(view_states, device)
            mask_batch = get_mask_batch(view_states, legal_batch, device)
            out = model(x_game_state, mask_batch, sample=True)

            htps = out.head_type_primaries.cpu().tolist()
            route_map = _build_route_map(mask_batch.route)

            for i in range(num_envs):
                htp = htps[i]
                action = out.get_action(i)
                sel_key = int(out.sel_keys[i].item())

                option_mask = None
                if PRIMARY_NUM_CHOICES[htp] > 0:
                    local = route_map[(htp, i)]
                    option_mask = mask_batch.option_masks[htp][local : local + 1].clone()
                sel_mask = None
                sel_gids = None
                if sel_key >= 0:
                    sel_mask = mask_batch.sel_masks[sel_key][i : i + 1].clone()
                    if mask_batch.sel_group_ids[sel_key] is not None:
                        sel_gids = mask_batch.sel_group_ids[sel_key][i : i + 1].clone()

                reward, done = env_mgr.step(i, action)

                transitions[i].append(
                    Transition(
                        x_game_state=_slice_x_game_state(x_game_state, i),
                        head_type_primary=htp,
                        option_index=int(out.option_indices[i].item()),
                        sel_key=sel_key,
                        selection_index=int(out.selection_indices[i].item()),
                        target_index=int(out.target_indices[i].item()),
                        retain_indices=out.retain_indices[i : i + 1].clone(),
                        option_mask=option_mask,
                        sel_mask=sel_mask,
                        sel_group_ids=sel_gids,
                        monster_alive_mask=mask_batch.monster_alive_mask[i : i + 1].clone(),
                        multipick_mask=mask_batch.multipick_mask[i : i + 1].clone(),
                        multipick_group_ids=mask_batch.multipick_group_ids[i : i + 1].clone(),
                        pick_num=int(mask_batch.pick_nums[i].item()),
                        log_prob_old=out.get_log_prob(i).detach(),
                        value=out.values[i],
                        reward=reward,
                        done=done,
                    )
                )
                ep_rewards[i] += reward
                ep_lengths[i] += 1
                if done:
                    completed.append(EpisodeStats(ep_rewards[i], ep_lengths[i]))
                    ep_rewards[i] = 0.0
                    ep_lengths[i] = 0

        view_states = env_mgr.get_view_states()
        legal_batch = env_mgr.get_legal_actions()
        x_game_state = encode_batch_game_state(view_states, device)
        mask_batch = get_mask_batch(view_states, legal_batch, device)
        out = model(x_game_state, mask_batch, sample=False)
        bootstrap_values = [out.values[i] for i in range(num_envs)]

    model.train()
    return transitions, bootstrap_values, completed


def _run_eval_episode(model: ActorCritic, device: torch.device) -> tuple[float, int]:
    env = slai.GameEnv(ascension=ASCENSION_LEVEL, fast_mode=FAST_MODE)
    obs = env.reset(seed=random.randint(0, 2**31 - 1))
    total_reward = 0.0
    length = 0
    terminated = False
    model.eval()
    with torch.no_grad():
        while not terminated:
            legal = env.get_legal_actions()
            if not legal:
                break
            x = encode_batch_game_state([obs], device)
            mb = get_mask_batch([obs], [legal], device)
            out = model(x, mb, sample=False)
            action = out.get_action(0)
            prev = obs
            obs, terminated = env.step(action)
            total_reward += compute_reward(prev, obs, terminated)
            length += 1
    model.train()
    return total_reward, length


# =============================================================================
# GAE and batch creation
# =============================================================================


def _compute_gae(rewards, values, dones, bootstrap_value, gamma, lam):
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


@dataclass
class TrajectoryBatch:
    transitions: list[Transition]
    log_probs_old: torch.Tensor
    values_old: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor

    def __len__(self) -> int:
        return len(self.transitions)


def _create_batch(transitions, bootstrap_values, gamma, lam, device) -> TrajectoryBatch:
    all_trans: list[Transition] = []
    all_returns: list[float] = []
    all_adv: list[float] = []
    for env_idx, env_trans in enumerate(transitions):
        if not env_trans:
            continue
        rewards = [t.reward for t in env_trans]
        values = [t.value for t in env_trans]
        dones = [t.done for t in env_trans]
        ret, adv = _compute_gae(rewards, values, dones, bootstrap_values[env_idx], gamma, lam)
        all_trans.extend(env_trans)
        all_returns.extend(ret)
        all_adv.extend(adv)

    log_probs_old = torch.stack([t.log_prob_old for t in all_trans]).to(device)
    values_old = torch.cat([t.value for t in all_trans], dim=0).detach().to(device)
    advantages = torch.tensor(all_adv, dtype=torch.float32, device=device).view(-1, 1)
    returns = torch.tensor(all_returns, dtype=torch.float32, device=device).view(-1, 1)
    if advantages.numel() > 1:
        advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-8)
    return TrajectoryBatch(all_trans, log_probs_old, values_old, returns, advantages)


# =============================================================================
# PPO update
# =============================================================================


def _minibatch_indices(total: int, minibatch_size: int) -> Iterator[list[int]]:
    indices = list(range(total))
    random.shuffle(indices)
    for i in range(0, total, minibatch_size):
        yield indices[i : i + minibatch_size]


def _update_ppo(
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
) -> tuple[float, float, float]:
    total_p = total_v = total_e = 0.0
    n = 0
    for _ in range(num_epochs):
        for mb in _minibatch_indices(len(batch), minibatch_size):
            mb_trans = [batch.transitions[i] for i in mb]
            x = _move_x_game_state(
                _concat_x_game_states([t.x_game_state for t in mb_trans]), device
            )
            mask_batch = _build_mask_batch_from_samples(mb_trans, device)
            rec = _rec_from_samples(mb_trans, device)

            log_probs_new, entropies, values_new = model.evaluate_actions(x, mask_batch, rec)

            mb_idx = torch.tensor(mb, dtype=torch.long, device=device)
            log_probs_old = batch.log_probs_old[mb_idx].to(device)
            advantages = torch.squeeze(batch.advantages[mb_idx], -1).to(device)
            returns = batch.returns[mb_idx].to(device)
            values_old = batch.values_old[mb_idx].to(device)

            ratio = torch.exp(log_probs_new - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            loss_policy = -torch.mean(torch.min(surr1, surr2))

            if clip_value_loss:
                values_clipped = values_old + torch.clamp(
                    values_new - values_old, -clip_eps, clip_eps
                )
                lv_unclipped = torch.pow(values_new - returns, 2)
                lv_clipped = torch.pow(values_clipped - returns, 2)
                loss_value = 0.5 * torch.mean(torch.max(lv_unclipped, lv_clipped))
            else:
                loss_value = F.mse_loss(values_new, returns)

            loss_entropy = -torch.mean(entropies)
            loss = loss_policy + coef_value * loss_value + coef_entropy * loss_entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_p += loss_policy.item()
            total_v += loss_value.item()
            total_e += loss_entropy.item()
            n += 1
    return total_p / n, total_v / n, total_e / n


# =============================================================================
# Training loop
# =============================================================================


def _get_entropy_schedule(num_iterations, elbow, max_coef, min_coef) -> list[float]:
    coefs = []
    slope = (min_coef - max_coef) / elbow
    for it in range(num_iterations):
        coefs.append(slope * it + max_coef if it <= elbow else min_coef)
    return coefs


def train(
    exp_name,
    num_iterations,
    log_every,
    save_every,
    model,
    optimizer,
    rollout_length,
    num_epochs,
    minibatch_size,
    clip_eps,
    clip_value_loss,
    gamma,
    lam,
    coef_value,
    coefs_entropy,
    max_grad_norm,
    num_envs,
    device,
) -> None:
    writer = SummaryWriter(f"experiments/{exp_name}")
    model.to(device)
    env_mgr = EnvironmentManager(num_envs)
    total_steps = 0

    try:
        for iteration in range(num_iterations):
            coef_entropy = coefs_entropy[iteration]
            transitions, bootstrap_values, completed = _collect_rollout(
                model, env_mgr, rollout_length, device
            )
            total_steps += rollout_length * num_envs
            batch = _create_batch(transitions, bootstrap_values, gamma, lam, device)
            if len(batch) == 0:
                print(f"Iteration {iteration}: Empty batch, skipping")
                continue
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
            if iteration % log_every == 0:
                print(
                    f"Iter {iteration} | steps={total_steps} | "
                    f"policy={loss_policy:.4f} value={loss_value:.4f} | episodes={len(completed)}"
                )
                writer.add_scalar("Loss/policy", loss_policy, iteration)
                writer.add_scalar("Loss/value", loss_value, iteration)
                writer.add_scalar("Loss/entropy", loss_entropy, iteration)
                writer.add_scalar("Entropy/coef", coef_entropy, iteration)
                writer.add_scalar("Steps/total", total_steps, iteration)
                if completed:
                    avg_r = sum(e.total_reward for e in completed) / len(completed)
                    avg_l = sum(e.length for e in completed) / len(completed)
                    writer.add_scalar("Episode/avg_reward", avg_r, iteration)
                    writer.add_scalar("Episode/avg_length", avg_l, iteration)
                    writer.add_scalar("Episode/completed_count", len(completed), iteration)
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
