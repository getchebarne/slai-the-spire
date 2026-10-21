import slai
import torch

from src.rl.algorithms.actor_critic.episode import EpisodeStats
from src.rl.algorithms.actor_critic.workers import Exchange
from src.rl.algorithms.actor_critic.workers import Worker
from src.rl.algorithms.actor_critic.workers import drain_episodes
from src.rl.algorithms.actor_critic.workers import snapshot_to_device
from src.rl.algorithms.actor_critic.workers import step_workers
from src.rl.models import ActorCritic
from src.rl.packed import cat_states
from src.rl.reward import RewardStream
from src.rl.types import X_GameState
from src.rl.types import X_Rollout


_AT_REST = int(slai.ActionType.Rest)

# The smith is argless: the deck selection it queues arrives as a separate halt
_AT_SMITH = int(slai.ActionType.RestSmith)


def collect_rollout(
    model: ActorCritic,
    exchange: Exchange,
    workers: list[Worker],
    rollout_length: int,
    device: torch.device,
    curriculum_c: float,
    generator: torch.Generator,
) -> tuple[X_Rollout, list[EpisodeStats], dict[str, float]]:
    """Play `rollout_length` lockstep steps of every game with `model`, sampling from
    `generator`: each step one forward over all games, then the workers step them. GAE is the
    learner's"""
    E = len(exchange.step.actions)
    T = rollout_length
    N = T * E
    K = len(RewardStream)
    exchange.curriculum_c.fill_(curriculum_c)
    chunks_x: list[X_GameState] = []  # per step, in state order
    chunks_count: list[torch.Tensor] = []  # their counts, on the CPU
    actions = torch.empty(N, 4, dtype=torch.long, device=device)
    log_probs = torch.empty(N, device=device)
    rewards = torch.empty(T, E, K, device=device)
    dones = torch.empty(T, E, device=device)

    # Rest-site preference: device-side counters, read once after the loop (no per-step sync)
    rest_selections = torch.zeros((), device=device)
    upgrade_selections = torch.zeros((), device=device)

    # Set model to eval, turn off gradient computation
    model.eval()
    with torch.no_grad():
        x, x_count_cpu = snapshot_to_device(exchange.step, device)
        for t in range(T):
            out = model.act(x, generator=generator)
            chunks_x.append(x)
            chunks_count.append(x_count_cpu)
            step_rows = slice(t * E, (t + 1) * E)
            actions[step_rows] = torch.stack(
                [out.action_type, out.slice_kind, out.rank, out.target], dim=1
            )
            log_probs[step_rows] = out.log_prob.sum(-1)

            # Campfire decision: both Rest and RestSmith legal — which was chosen?
            both = x.x_legal_action_type[:, _AT_REST] & x.x_legal_action_type[:, _AT_SMITH]
            chosen = out.action_type
            rest_selections.add_((both & (chosen == _AT_REST)).sum())
            upgrade_selections.add_((both & (chosen == _AT_SMITH)).sum())

            exchange.step.actions.copy_(actions[step_rows])
            step_workers(workers)
            rewards[t] = exchange.step.rewards
            dones[t] = exchange.step.dones
            x, x_count_cpu = snapshot_to_device(exchange.step, device)

    # Set model to training mode
    model.train()

    # Surface the termination vs rollout-boundary-bootstrap split (the GAE truncation
    # path): how many envs were still mid-episode at T and relied on V(s_T).
    n_campfire_selections = (rest_selections + upgrade_selections).item()
    rollout_info = {
        "rollout/terminations": dones.sum().item(),
        "rollout/boundary_truncations": float(E) - dones[T - 1].sum().item(),
        # 0.0 also when no campfire was seen this rollout
        "rollout/rest_upgrade_rate": upgrade_selections.item() / n_campfire_selections
        if n_campfire_selections
        else 0.0,
    }

    return (
        X_Rollout(
            game_state=cat_states(chunks_x),
            x_count_cpu=torch.cat(chunks_count),
            actions=actions,
            log_probs_behavior=log_probs,
            rewards=rewards,
            dones=dones,
            final_game_state=x,
        ),
        drain_episodes(exchange.episodes),
        rollout_info,
    )
