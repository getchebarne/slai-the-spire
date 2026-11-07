import multiprocessing as mp

import pytest
import torch

from src.rl.algorithms.actor_critic import envs
from src.rl.algorithms.actor_critic.workers import pool
from src.rl.algorithms.actor_critic.workers import snapshot_to_device
from src.rl.algorithms.actor_critic.workers import start_workers
from src.rl.algorithms.actor_critic.workers import step_workers
from src.rl.algorithms.actor_critic.workers import stop_workers
from src.rl.schema.context import ASCENSION
from src.rl.tensor_ops import iter_tensors
from src.rl.types import TokenKind


_worker_loop = pool._worker_loop


def _short_episode_worker(*args) -> None:
    """Run a real worker with episodes truncated after two policy steps.

    Parameters
    ----------
    args : tuple
        The arguments passed to the original worker loop.
    """
    envs._MAX_STEPS_EPISODE = 2
    _worker_loop(*args)


def test_workers_own_results_and_publish_episodes_on_their_step(monkeypatch):
    """Exercise stepping, reset curriculum, owned outputs and shutdown across two workers.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces the spawn target with the short-episode wrapper for this test.
    """
    monkeypatch.setattr(pool, "_worker_loop", _short_episode_worker)
    child_pids = {process.pid for process in mp.active_children()}
    workers = start_workers(4, 2, 0.0, 0.99, 9)
    try:
        assert len({process.pid for process in mp.active_children()} - child_pids) == 2
        batch, _ = snapshot_to_device(workers, torch.device("cpu"))
        retained = []
        with pytest.raises(ValueError, match="shape"):
            step_workers(workers, torch.zeros(4, 2, dtype=torch.long), 1.5)
        with pytest.raises(ValueError, match="dtype"):
            step_workers(workers, torch.zeros(4, 3), 1.5)

        for step in range(4):
            retained.extend((x_tensor, x_tensor.clone()) for x_tensor in iter_tensors(batch))

            # Choose one action from each state's public masks, including its Monster target.
            x_actions = torch.full((4, 3), -1, dtype=torch.long)
            tokens = batch.tokens.rows
            start = 0
            for state, length in enumerate(batch.tokens.x_lengths.tolist()):
                stop = start + length
                action_type = int(batch.x_legal_action_type[state].nonzero()[0, 0])
                x_selections = tokens.x_legal_selection[start:stop, action_type].nonzero()
                selection = int(x_selections[0, 0]) if len(x_selections) else -1
                target = -1
                if selection >= 0 and bool(tokens.x_needs_target[start + selection]):
                    x_monsters = tokens.x_kind[start:stop] == TokenKind.MONSTERS
                    x_targets = tokens.x_legal_target[start:stop, action_type][x_monsters]
                    target = int(x_targets.nonzero()[0, 0])
                x_actions[state] = torch.tensor([action_type, selection, target])
                start = stop

            x_rewards, x_dones, episodes = step_workers(
                workers, x_actions, 1.5 if step < 3 else 2.5
            )
            assert x_rewards.device.type == x_dones.device.type == "cpu"
            assert bool(torch.isfinite(x_rewards).all())
            assert x_dones.tolist() == [float(step % 2 == 1)] * 4
            assert len(episodes) == (4 if step % 2 else 0)
            assert all(episode.length == 2 and episode.truncated for episode in episodes)
            retained.extend((x_tensor, x_tensor.clone()) for x_tensor in (x_rewards, x_dones))

            # Curriculum changes affect the reset game; earlier snapshots and results stay owned.
            batch, _ = snapshot_to_device(workers, torch.device("cpu"))
            expected_ascensions = (
                [0, 0, 0, 0] if step == 0 else [3, 3, 2, 2] if step == 3 else [2, 2, 1, 1]
            )
            assert batch.x_context[:, ASCENSION].sum(-1).tolist() == expected_ascensions
            for x_actual, x_expected in retained:
                torch.testing.assert_close(x_actual, x_expected, rtol=0, atol=0)
    finally:
        stop_workers(workers)
        stop_workers(workers)

    assert {process.pid for process in mp.active_children()} == child_pids
    with pytest.raises(RuntimeError, match="stopped"):
        step_workers(workers, torch.zeros(4, 3, dtype=torch.long), 0.0)
    with pytest.raises(RuntimeError, match="stopped"):
        snapshot_to_device(workers, torch.device("cpu"))
