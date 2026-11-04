import slai
import torch

from src.rl.algorithms.actor_critic.curriculum import ascensions
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


_ACTION_TYPE_REST = int(slai.ActionType.Rest)

# The smith is argless: the deck selection it queues arrives as a separate halt
_ACTION_TYPE_SMITH = int(slai.ActionType.RestSmith)


def collect_rollout(
    model: ActorCritic,
    exchange: Exchange,
    workers: list[Worker],
    rollout_length: int,
    device: torch.device,
    ascension_current: float,
    generator: torch.Generator,
) -> tuple[X_Rollout, list[EpisodeStats], dict[str, float]]:
    """Play `rollout_length` lockstep steps of every game with `model`, sampling from `generator`.

    Each step runs one forward over all games, then the workers step them. GAE is the learner's.

    Parameters
    ----------
    model : ActorCritic
        The actor's model, in eval mode during the rollout and back in training mode after it.
    exchange : Exchange
        What the step loop and its workers share; its step holds the E games.
    workers : list[Worker]
        The worker processes that step the games.
    rollout_length : int
        The steps to play, T.
    device : torch.device
        The model's device.
    ascension_current : float
        The soft-curriculum position, which sets the ascension of a game that resets.
    generator : torch.Generator
        The generator the actions are sampled from.

    Returns
    -------
    x_rollout : X_Rollout
        The T steps of the E games, before GAE.
    episodes : list[EpisodeStats]
        The finished episodes, drained from the workers' queue.
    metrics : dict[str, float]
        The rollout's scalars: its terminations (step-limit cuts included), boundary truncations and
        Rest Site upgrade rate.
    """
    E = len(exchange.x_step.x_actions)
    T = rollout_length
    N = T * E
    K = len(RewardStream)
    exchange.x_ascensions.copy_(torch.tensor(ascensions(ascension_current, E)))
    x_gss: list[X_GameState] = []  # Per step, in state order
    x_count_cpus: list[torch.Tensor] = []  # Their counts, on the CPU
    x_actions = torch.empty(N, 4, dtype=torch.long, device=device)
    x_log_probs_behavior = torch.empty(N, device=device)
    x_rewards = torch.empty(T, E, K, device=device)
    x_dones = torch.empty(T, E, device=device)

    # Rest-site preference: device-side counters, read once after the loop (no per-step sync)
    x_rest_selections = torch.zeros((), device=device)
    x_upgrade_selections = torch.zeros((), device=device)

    # Set model to eval, turn off gradient computation
    model.eval()
    with torch.no_grad():
        x_gs, x_count_cpu = snapshot_to_device(exchange.x_step, device)
        for t in range(T):
            x_action = model.act(x_gs, generator=generator)
            x_gss.append(x_gs)
            x_count_cpus.append(x_count_cpu)
            step_rows = slice(t * E, (t + 1) * E)
            x_actions[step_rows] = torch.stack(
                [
                    x_action.x_action_type,
                    x_action.x_token_kind,
                    x_action.x_rank,
                    x_action.x_target,
                ],
                dim=1,
            )
            x_log_probs_behavior[step_rows] = x_action.x_log_prob.sum(-1)

            # Campfire decision: both Rest and RestSmith legal — which was chosen?
            x_both = (
                x_gs.x_legal_action_type[:, _ACTION_TYPE_REST]
                & x_gs.x_legal_action_type[:, _ACTION_TYPE_SMITH]
            )
            x_rest_selections.add_((x_both & (x_action.x_action_type == _ACTION_TYPE_REST)).sum())
            x_upgrade_selections.add_(
                (x_both & (x_action.x_action_type == _ACTION_TYPE_SMITH)).sum()
            )

            exchange.x_step.x_actions.copy_(x_actions[step_rows])
            step_workers(workers)
            x_rewards[t] = exchange.x_step.x_rewards
            x_dones[t] = exchange.x_step.x_dones
            x_gs, x_count_cpu = snapshot_to_device(exchange.x_step, device)

    # Set model to training mode
    model.train()

    # Surface the termination vs rollout-boundary-bootstrap split (the GAE truncation
    # path): how many envs were still mid-episode at T and relied on V(s_T).
    n_campfire_selections = (x_rest_selections + x_upgrade_selections).item()
    metrics = {
        "rollout/terminations": x_dones.sum().item(),
        "rollout/boundary_truncations": float(E) - x_dones[T - 1].sum().item(),
        # 0.0 also when no campfire was seen this rollout
        "rollout/rest_upgrade_rate": x_upgrade_selections.item() / n_campfire_selections
        if n_campfire_selections
        else 0.0,
    }

    return (
        X_Rollout(
            x_gs=cat_states(x_gss),
            x_count_cpu=torch.cat(x_count_cpus),
            x_actions=x_actions,
            x_log_probs_behavior=x_log_probs_behavior,
            x_rewards=x_rewards,
            x_dones=x_dones,
            x_gs_final=x_gs,
        ),
        drain_episodes(exchange.episodes),
        metrics,
    )
