import copy
import multiprocessing as mp

import pytest
import torch

from src.rl.algorithms.actor_critic.actor import RolloutConfig
from src.rl.algorithms.actor_critic.actor import publish
from src.rl.algorithms.actor_critic.actor import receive_rollout
from src.rl.algorithms.actor_critic.actor import start_actor
from src.rl.algorithms.actor_critic.actor import stop_actor
from src.rl.models import ActorCritic
from src.rl.tensor_ops import iter_tensors


_MODEL_CONFIG = dict(
    dim_entity=32,
    dim_global=64,
    transformer_dim_ff=64,
    transformer_num_heads=4,
    transformer_num_blocks=1,
    gnn_num_layers=1,
    map_encoder_dim=16,
    dim_ff_primary=64,
    dim_ff_value=64,
    dim_op=32,
    dim_key=32,
)


def test_actor_publishes_weights_one_update_behind_and_owns_rollouts():
    original_threads = torch.get_num_threads()
    child_pids = {process.pid for process in mp.active_children()}
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(41)
            model = ActorCritic(**_MODEL_CONFIG).eval()
            initial_model = copy.deepcopy(model)
            x_rng_before = torch.get_rng_state().clone()
            actor = start_actor(
                model, _MODEL_CONFIG, RolloutConfig(2, 2, 2), torch.device("cpu"), 0.99
            )
            torch.testing.assert_close(torch.get_rng_state(), x_rng_before, rtol=0, atol=0)
            try:
                publish(actor, model, 0, 0.0)
                first, seconds_wait = receive_rollout(actor)
                assert seconds_wait >= 0
                retained = [
                    (x_tensor, x_tensor.clone()) for x_tensor in iter_tensors(first.rollout)
                ]

                # The next rollout still uses version 0; the following one uses this publication.
                with torch.no_grad():
                    for x_parameter in model.parameters():
                        x_parameter.add_(torch.randn_like(x_parameter) * 0.05)
                publish(actor, model, 1, 1.5)
                second, _ = receive_rollout(actor)
                third, _ = receive_rollout(actor)
                with torch.no_grad():
                    for delivery, expected_model in (
                        (first, initial_model),
                        (second, initial_model),
                        (third, model),
                    ):
                        rollout = delivery.rollout
                        assert rollout.x_actions.shape == (4, 3)
                        assert rollout.x_rewards.shape[:2] == (2, 2)
                        x_log_prob, *_ = expected_model.evaluate_actions(
                            rollout.states, rollout.x_actions
                        )
                        torch.testing.assert_close(
                            x_log_prob, rollout.x_log_probs_behavior, rtol=1e-5, atol=1e-5
                        )
                    x_old_log_prob, *_ = initial_model.evaluate_actions(
                        third.rollout.states, third.rollout.x_actions
                    )
                    assert not torch.allclose(x_old_log_prob, third.rollout.x_log_probs_behavior)
                for x_actual, x_expected in retained:
                    torch.testing.assert_close(x_actual, x_expected, rtol=0, atol=0)
            finally:
                stop_actor(actor)
                stop_actor(actor)

            with pytest.raises(RuntimeError, match="stopped"):
                publish(actor, model, 2, 1.5)
            with pytest.raises(RuntimeError, match="stopped"):
                receive_rollout(actor)
        assert {process.pid for process in mp.active_children()} == child_pids
    finally:
        torch.set_num_threads(original_threads)


def test_actor_failure_surfaces_through_public_operations():
    original_threads = torch.get_num_threads()
    child_pids = {process.pid for process in mp.active_children()}
    torch.set_num_threads(1)
    try:
        model = ActorCritic(**_MODEL_CONFIG)
        actor = start_actor(
            model,
            {**_MODEL_CONFIG, "transformer_num_heads": 0},
            RolloutConfig(2, 2, 2),
            torch.device("cpu"),
            0.99,
        )
        try:
            with pytest.raises(RuntimeError, match="actor process died"):
                publish(actor, model, 0, 0.0)
                receive_rollout(actor)
        finally:
            stop_actor(actor)
        assert {process.pid for process in mp.active_children()} == child_pids
    finally:
        torch.set_num_threads(original_threads)
