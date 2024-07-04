from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.game.combat.view import CombatView


@dataclass
class Batch:
    state_ts: list[CombatView]
    state_tp1s: list[CombatView]
    actions: list[int]
    rewards: list[float]
    valid_action_mask_tp1s: list[list[bool]]
    game_over_flags: list[bool]


@dataclass
class Sample:
    state_t: CombatView
    state_tp1: CombatView
    action: int
    reward: float
    valid_action_mask_tp1: list[bool]
    game_over_flag: bool


class ReplayBuffer:
    def __init__(self, size: int):
        self.size = size

        self.state_ts: list[CombatView] = [None] * size
        self.state_tp1s: list[CombatView] = [None] * size
        self.actions: list[int] = [None] * size
        self.rewards: list[float] = [None] * size
        self.valid_action_mask_tp1s: list[CombatView] = [None] * size
        self.game_over_flags: list[CombatView] = [None] * size

    def __len__(self) -> int:
        return self.size


def store(replay_buffer: ReplayBuffer, sample: Sample, index: int) -> int:
    replay_buffer.state_ts[index] = sample.state_t
    replay_buffer.state_tp1s[index] = sample.state_tp1
    replay_buffer.actions[index] = sample.action
    replay_buffer.rewards[index] = sample.reward
    replay_buffer.valid_action_mask_tp1s[index] = sample.valid_action_mask_tp1
    replay_buffer.game_over_flags[index] = sample.game_over_flag

    return (index + 1) % len(replay_buffer)


def sample(replay_buffer: ReplayBuffer, batch_size: int, up_to: int) -> Optional[Batch]:

    # TODO: should happen outside of this function
    # if not full and ptr < batch_size:
    #     return None

    rng = np.random.default_rng()  # TODO: seed
    sample_indexes = rng.integers(low=0, high=up_to, size=batch_size)

    # Initialize empty batch
    batch = Batch(
        state_ts=[None] * batch_size,
        state_tp1s=[None] * batch_size,
        actions=[None] * batch_size,
        rewards=[None] * batch_size,
        valid_action_mask_tp1s=[None] * batch_size,
        game_over_flags=[None] * batch_size,
    )

    # Fill it
    for batch_index, sample_index in enumerate(sample_indexes):
        batch.state_ts[batch_index] = replay_buffer.state_ts[sample_index]
        batch.state_tp1s[batch_index] = replay_buffer.state_tp1s[sample_index]
        batch.actions[batch_index] = replay_buffer.actions[sample_index]
        batch.rewards[batch_index] = replay_buffer.rewards[sample_index]
        batch.valid_action_mask_tp1s[batch_index] = replay_buffer.valid_action_mask_tp1s[
            sample_index
        ]
        batch.game_over_flags[batch_index] = replay_buffer.game_over_flags[sample_index]

    return batch
