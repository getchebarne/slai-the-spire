"""The run's metrics: their keys, the episode aggregation and the TensorBoard logging."""

from enum import StrEnum

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.rl.algorithms.actor_critic.actor import Delivery
from src.rl.algorithms.actor_critic.envs import EpisodeResult
from src.rl.reward import RewardStream
from src.rl.types import RunState


class MetricKey(StrEnum):
    """A metric's key; a `_MetricPrefix` in front of it is its TensorBoard name.

    The code that computes a metric and the code that reads it share the key, so producers and
    consumers can't drift apart.
    """

    ACT2_RATE = "act2_rate"
    ACT3_RATE = "act3_rate"
    ACT4_RATE = "act4_rate"
    ASCENSION = "ascension"
    BOSS_RATE = "boss_rate"
    COMPLETED_COUNT = "completed_count"
    DECK_SIZE_AVG = "deck_size_avg"
    ENTROPY_L1 = "entropy/l1"
    ENTROPY_L2 = "entropy/l2"
    ENTROPY_L3 = "entropy/l3"
    FLOOR_AVG = "floor_avg"
    GOLD_AVG = "gold_avg"
    GOLD_SPENT_AVG = "gold_spent_avg"
    GRAD_NORM_PRECLIP = "grad/norm_preclip"
    HEART_DAMAGE_AVG = "heart_damage_avg"
    KEYS_AT_DEATH_AVG = "keys_at_death_avg"
    KL_UNIFORM_L1 = "kl_uniform/l1"
    KL_UNIFORM_L2 = "kl_uniform/l2"
    KL_UNIFORM_L3 = "kl_uniform/l3"
    LENGTH_AVG = "length_avg"
    LOSS_ENTROPY = "loss/entropy"
    LOSS_KL_UNIFORM = "loss/kl_uniform"
    LOSS_POLICY = "loss/policy"
    LOSS_VALUE = "loss/value"
    MEMORY_ACTOR_GB = "memory/actor_gb"
    MEMORY_LEARNER_GB = "memory/learner_gb"
    OFFPOLICY_IS_WEIGHT_MEAN = "offpolicy/is_weight_mean"
    OFFPOLICY_IS_WEIGHT_TRUNCATED = "offpolicy/is_weight_truncated"
    OFFPOLICY_KL_BEHAVIOR_PROX = "offpolicy/kl_behavior_prox"
    PIPELINE_ACTOR_WAIT = "pipeline/actor_wait"
    PIPELINE_LEARNER_WAIT = "pipeline/learner_wait"
    PIPELINE_SAMPLES_PER_S = "pipeline/samples_per_s"
    RETURN_AVG = "return_avg"  # Also each reward stream's: return_avg_<stream>
    RETURN_STD = "return_std"
    STEPS_TOTAL = "steps/total"
    TIME_ROLLOUT = "time/rollout"
    TIME_UPDATE = "time/update"
    TRUNCATED_RATE = "truncated_rate"
    UPDATE_APPROX_KL = "update/approx_kl"
    UPDATE_CLIP_FRACTION = "update/clip_fraction"
    UPDATE_EPOCHS_RUN = "update/epochs_run"
    UPDATE_LR_SCALE = "update/lr_scale"
    VALUE_EV = "value/ev"  # Each reward stream's: value/ev_<stream>
    WIN_RATE = "win_rate"


class _MetricPrefix(StrEnum):
    """The TensorBoard group a metric is logged under: the training run's or the evals'."""

    EVAL = "eval/"
    TRAIN = "train/"


def log_iteration(
    writer: SummaryWriter,
    iteration: int,
    run_state: RunState,
    metrics: dict[str, float],
    episode_results: list[EpisodeResult],
    delivery: Delivery,
    seconds_update: float,
    seconds_wait_learner: float,
    samples_per_s: float,
    device: torch.device,
) -> None:
    """Write an iteration's metrics of each kind under train/ and print its line.

    Parameters
    ----------
    writer : SummaryWriter
        Rank 0's writer.
    iteration : int
        The iteration.
    run_state : RunState
        The run's progress after the iteration.
    metrics : dict[str, float]
        The update's metrics, keyed by `MetricKey`.
    episode_results : list[EpisodeResult]
        The episodes the iteration's rollouts finished, every rank's, possibly none.
    delivery : Delivery
        The iteration's rollout, with the Actor's timings and memory peak.
    seconds_update : float
        How long the update took.
    seconds_wait_learner : float
        How long the Learner waited for the rollout.
    samples_per_s : float
        The steps trained on per second of wall clock, over the iteration.
    device : torch.device
        The Learner's device.
    """
    _log_training(writer, iteration, metrics, run_state.lr_scale)
    _log_behavior(
        writer, iteration, _MetricPrefix.TRAIN, episode_results, run_state.ascension_current
    )
    _log_misc(
        writer,
        iteration,
        run_state.total_steps,
        delivery,
        seconds_update,
        seconds_wait_learner,
        samples_per_s,
        device,
    )
    _print_iteration(iteration, run_state.total_steps, metrics, len(episode_results))


def log_eval(
    writer: SummaryWriter,
    iteration_eval: int,
    ascension_current_eval: float,
    episode_results_eval: list[EpisodeResult],
) -> dict[str, float]:
    """Write a finished eval battery's behavioral metrics under eval/ and print its line.

    Parameters
    ----------
    writer : SummaryWriter
        Rank 0's writer.
    iteration_eval : int
        The iteration whose weights the battery played.
    ascension_current_eval : float
        The curriculum position it played at.
    episode_results_eval : list[EpisodeResult]
        The battery's finished episodes.

    Returns
    -------
    dict[str, float]
        The battery's behavioral metrics, keyed by `MetricKey`.
    """
    metrics_eval = _log_behavior(
        writer, iteration_eval, _MetricPrefix.EVAL, episode_results_eval, ascension_current_eval
    )
    _print_eval(iteration_eval, ascension_current_eval, metrics_eval)
    return metrics_eval


def _log_training(
    writer: SummaryWriter, iteration: int, metrics: dict[str, float], lr_scale: float
) -> None:
    """Write an update's training metrics under train/, with the learning-rate scale it left.

    Parameters
    ----------
    writer : SummaryWriter
        Rank 0's writer.
    iteration : int
        The iteration.
    metrics : dict[str, float]
        The update's metrics, keyed by `MetricKey`.
    lr_scale : float
        The adaptive-KL learning-rate scale after the update.
    """
    _write_metrics(
        writer, iteration, _MetricPrefix.TRAIN, {**metrics, MetricKey.UPDATE_LR_SCALE: lr_scale}
    )


def _log_behavior(
    writer: SummaryWriter,
    iteration: int,
    metric_prefix: _MetricPrefix,
    episode_results: list[EpisodeResult],
    ascension_current: float,
) -> dict[str, float]:
    """Aggregate finished episodes into behavioral metrics, write them and return them.

    Parameters
    ----------
    writer : SummaryWriter
        Rank 0's writer.
    iteration : int
        The iteration the metrics are logged at.
    metric_prefix : _MetricPrefix
        Whose episodes these are: the training rollouts' or an eval battery's.
    episode_results : list[EpisodeResult]
        The finished episodes, possibly none.
    ascension_current : float
        The curriculum position they were played at.

    Returns
    -------
    dict[str, float]
        The metrics written, keyed by `MetricKey`; only the ascension without an episode.
    """
    metrics = _aggregate_episode_results(episode_results) if episode_results else {}
    metrics[MetricKey.ASCENSION] = ascension_current
    _write_metrics(writer, iteration, metric_prefix, metrics)
    return metrics


def _log_misc(
    writer: SummaryWriter,
    iteration: int,
    total_steps: int,
    delivery: Delivery,
    seconds_update: float,
    seconds_wait_learner: float,
    samples_per_s: float,
    device: torch.device,
) -> None:
    """Write an iteration's progress, timings and memory peaks under train/.

    Parameters
    ----------
    writer : SummaryWriter
        Rank 0's writer.
    iteration : int
        The iteration.
    total_steps : int
        The steps trained on so far, every rank's counted.
    delivery : Delivery
        The iteration's rollout, with the Actor's timings and memory peak.
    seconds_update : float
        How long the update took.
    seconds_wait_learner : float
        How long the Learner waited for the rollout.
    samples_per_s : float
        The steps trained on per second of wall clock, over the iteration.
    device : torch.device
        The Learner's device.
    """
    metrics: dict[str, float] = {
        MetricKey.STEPS_TOTAL: total_steps,
        MetricKey.TIME_ROLLOUT: delivery.seconds_collect,
        MetricKey.TIME_UPDATE: seconds_update,
        MetricKey.PIPELINE_ACTOR_WAIT: delivery.seconds_wait_actor,
        MetricKey.PIPELINE_LEARNER_WAIT: seconds_wait_learner,
        MetricKey.PIPELINE_SAMPLES_PER_S: samples_per_s,
    }

    # Both processes' peak GPU memory
    if device.type == "cuda":
        metrics[MetricKey.MEMORY_ACTOR_GB] = delivery.bytes_reserved_peak / 2**30
        metrics[MetricKey.MEMORY_LEARNER_GB] = torch.cuda.max_memory_reserved(device) / 2**30
    _write_metrics(writer, iteration, _MetricPrefix.TRAIN, metrics)


def _print_iteration(
    iteration: int, total_steps: int, metrics: dict[str, float], num_episodes: int
) -> None:
    """Print an iteration's line.

    Parameters
    ----------
    iteration : int
        The iteration.
    total_steps : int
        The steps trained on so far, every rank's counted.
    metrics : dict[str, float]
        The update's metrics, keyed by `MetricKey`.
    num_episodes : int
        The episodes its rollouts finished, every rank's counted.
    """
    print(
        f"Iter {iteration} | steps={total_steps} | "
        f"policy={metrics[MetricKey.LOSS_POLICY]:.4f} "
        f"value={metrics[MetricKey.LOSS_VALUE]:.4f} | "
        f"episodes={num_episodes}"
    )


def _print_eval(
    iteration_eval: int, ascension_current_eval: float, metrics_eval: dict[str, float]
) -> None:
    """Print a finished eval battery's line.

    Parameters
    ----------
    iteration_eval : int
        The iteration whose weights the battery played.
    ascension_current_eval : float
        The curriculum position it played at.
    metrics_eval : dict[str, float]
        Its behavioral metrics, keyed by `MetricKey`.
    """
    print(
        f"  eval@{iteration_eval} c={ascension_current_eval:.2f}: "
        f"win_rate={metrics_eval[MetricKey.WIN_RATE]:.3f} "
        f"floor={metrics_eval[MetricKey.FLOOR_AVG]:.2f} "
        f"act2={metrics_eval[MetricKey.ACT2_RATE]:.2f} "
        f"act3={metrics_eval[MetricKey.ACT3_RATE]:.2f} "
        f"act4={metrics_eval[MetricKey.ACT4_RATE]:.2f} "
        f"return={metrics_eval[MetricKey.RETURN_AVG]:.4f}"
    )


def _write_metrics(
    writer: SummaryWriter, iteration: int, metric_prefix: _MetricPrefix, metrics: dict[str, float]
) -> None:
    """Write metrics at an iteration, each under `metric_prefix` plus its key.

    Parameters
    ----------
    writer : SummaryWriter
        Rank 0's writer.
    iteration : int
        The iteration.
    metric_prefix : _MetricPrefix
        The group to write them under.
    metrics : dict[str, float]
        The metrics, keyed by `MetricKey`.
    """
    for key, value in metrics.items():
        writer.add_scalar(metric_prefix + key, value, iteration)


def _aggregate_episode_results(episode_results: list[EpisodeResult]) -> dict[str, float]:
    """Compute outcome metrics over a batch of episode results.

    Parameters
    ----------
    episode_results : list[EpisodeResult]
        The episode_results, at least one.

    Returns
    -------
    dict[str, float]
        The metrics, keyed by `MetricKey`, with the episode count and each reward stream's
        discounted return; the keys' only when a game died, and the Heart's only when a game got
        there.
    """
    num_episodes = len(episode_results)
    returns = [float(episode.stream_returns.sum()) for episode in episode_results]
    keys_at_death = [
        episode.keys_at_death for episode in episode_results if episode.keys_at_death is not None
    ]
    heart_damage = [
        episode.heart_damage for episode in episode_results if episode.heart_damage is not None
    ]
    metrics: dict[str, float] = {
        MetricKey.WIN_RATE: sum(episode.won for episode in episode_results) / num_episodes,
        MetricKey.FLOOR_AVG: sum(episode.floor for episode in episode_results) / num_episodes,
        MetricKey.LENGTH_AVG: sum(episode.length for episode in episode_results) / num_episodes,
        MetricKey.RETURN_AVG: float(np.mean(returns)),
        MetricKey.RETURN_STD: float(np.std(returns)),
        MetricKey.ACT2_RATE: sum(episode.act >= 2 for episode in episode_results) / num_episodes,
        MetricKey.ACT3_RATE: sum(episode.act >= 3 for episode in episode_results) / num_episodes,
        MetricKey.ACT4_RATE: sum(episode.act >= 4 for episode in episode_results) / num_episodes,
        MetricKey.GOLD_AVG: sum(episode.gold_gained for episode in episode_results) / num_episodes,
        MetricKey.GOLD_SPENT_AVG: sum(episode.gold_spent for episode in episode_results) / num_episodes,
        MetricKey.BOSS_RATE: sum(episode.boss_reached for episode in episode_results) / num_episodes,
        MetricKey.DECK_SIZE_AVG: sum(episode.deck_size for episode in episode_results) / num_episodes,
        MetricKey.TRUNCATED_RATE: sum(episode.truncated for episode in episode_results) / num_episodes,
    }

    # Over the games that died, and over those that got to the Heart
    if keys_at_death:
        metrics[MetricKey.KEYS_AT_DEATH_AVG] = sum(keys_at_death) / len(keys_at_death)
    if heart_damage:
        metrics[MetricKey.HEART_DAMAGE_AVG] = sum(heart_damage) / len(heart_damage)
    metrics[MetricKey.COMPLETED_COUNT] = num_episodes
    for k, name in enumerate(RewardStream):
        # Discounted per-stream returns, the optimizer's actual objective:
        # undiscounted sums shift with episode length
        metrics[f"{MetricKey.RETURN_AVG}_{name}"] = (
            sum(episode.stream_returns[k] for episode in episode_results) / num_episodes
        )

    return metrics
