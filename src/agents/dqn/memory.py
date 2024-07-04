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
        self._size = size

        self._state_ts: list[CombatView] = [None] * size
        self._state_tp1s: list[CombatView] = [None] * size
        self._actions: list[int] = [None] * size
        self._rewards: list[float] = [None] * size
        self._valid_action_mask_tp1s: list[CombatView] = [None] * size
        self._game_over_flags: list[CombatView] = [None] * size

        # Current write index
        self._index: int = 0

        # Whether the buffer is full or not
        self._full: bool = False

        # RNG for sampling TODO: seed
        self._rng = np.random.default_rng()

    @property
    def full(self) -> bool:
        return self._full

    def __len__(self) -> int:
        return self._size

    def store(self, sample: Sample) -> None:
        self._state_ts[self._index] = sample.state_t
        self._state_tp1s[self._index] = sample.state_tp1
        self._actions[self._index] = sample.action
        self._rewards[self._index] = sample.reward
        self._valid_action_mask_tp1s[self._index] = sample.valid_action_mask_tp1
        self._game_over_flags[self._index] = sample.game_over_flag

        # Increment index
        self._index = (self._index + 1) % len(self)

        # Set full if needed
        self._full = self._full or self._index == 0

    def sample(self, batch_size: int) -> Optional[Batch]:
        if self._index < batch_size and not self._full:
            return

        high = len(self) if self._full else self._index
        sample_indexes = self._rng.integers(low=0, high=high, size=batch_size)

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
            batch.state_ts[batch_index] = self._state_ts[sample_index]
            batch.state_tp1s[batch_index] = self._state_tp1s[sample_index]
            batch.actions[batch_index] = self._actions[sample_index]
            batch.rewards[batch_index] = self._rewards[sample_index]
            batch.valid_action_mask_tp1s[batch_index] = self._valid_action_mask_tp1s[sample_index]
            batch.game_over_flags[batch_index] = self._game_over_flags[sample_index]

        return batch
