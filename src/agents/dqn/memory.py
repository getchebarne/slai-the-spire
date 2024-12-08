from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from src.game.combat.constant import MAX_HAND_SIZE
from src.game.combat.constant import MAX_MONSTERS


@dataclass
class Batch:
    state_ts: torch.Tensor
    state_tp1s: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    valid_action_mask_tp1s: torch.Tensor
    game_over_flags: torch.Tensor


@dataclass
class Sample:
    state_t: torch.Tensor
    state_tp1: torch.Tensor
    action: int
    reward: float
    valid_action_mask_tp1: torch.Tensor
    game_over_flag: bool


class ReplayBuffer:
    def __init__(self, size: int):
        self._size = size

        # Initially, these are None because we don't know the dimensions yet
        self._state_ts = None
        self._state_tp1s = None

        # Preallocate tensors with fixed shapes for scalars
        self._actions = torch.zeros(size, dtype=torch.long)
        self._rewards = torch.zeros(size, dtype=torch.float32)
        self._valid_action_mask_tp1s = torch.zeros(
            (size, MAX_HAND_SIZE + MAX_MONSTERS + 1), dtype=torch.float32
        )
        self._game_over_flags = torch.zeros(size, dtype=torch.float32)

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
        # Initialize tensors for state if not already initialized
        if self._state_ts is None:
            state_shape = sample.state_t.shape

            self._state_ts = torch.zeros((self._size, *state_shape), dtype=sample.state_t.dtype)
            self._state_tp1s = torch.zeros(
                (self._size, *state_shape), dtype=sample.state_tp1.dtype
            )

        # Store the data in the appropriate location
        self._state_ts[self._index] = sample.state_t
        self._state_tp1s[self._index] = sample.state_tp1
        self._actions[self._index] = sample.action
        self._rewards[self._index] = sample.reward
        self._valid_action_mask_tp1s[self._index] = sample.valid_action_mask_tp1
        self._game_over_flags[self._index] = sample.game_over_flag

        # Increment index
        self._index = (self._index + 1) % self._size

        # Set full if needed
        self._full = self._full or self._index == 0

    def sample(self, batch_size: int) -> Optional[Batch]:
        if self._index < batch_size and not self._full:
            return None

        high = self._size if self._full else self._index
        sample_indexes = self._rng.integers(low=0, high=high, size=batch_size)

        # Gather sampled data
        return Batch(
            state_ts=self._state_ts[sample_indexes],
            state_tp1s=self._state_tp1s[sample_indexes],
            actions=self._actions[sample_indexes],
            rewards=self._rewards[sample_indexes],
            valid_action_mask_tp1s=self._valid_action_mask_tp1s[sample_indexes],
            game_over_flags=self._game_over_flags[sample_indexes],
        )
