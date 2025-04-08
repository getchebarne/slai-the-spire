from dataclasses import dataclass

import numpy as np
import torch

from src.game.combat.constant import MAX_SIZE_HAND


@dataclass
class Batch:
    states: torch.Tensor
    states_next: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    valid_action_masks_next: torch.Tensor
    game_over_flags: torch.Tensor


@dataclass
class Sample:
    state_t: torch.Tensor
    state_tp1: torch.Tensor
    action: int
    reward: float
    valid_action_mask_next: torch.Tensor
    game_over_flag: bool


class ReplayBuffer:
    def __init__(self, size: int, seed: int = 42):
        self._size = size

        # These are `None` initially because we don't know their dimensions yet
        self._states = None
        self._states_next = None

        # Preallocate tensors with fixed shapes for scalars
        self._actions = torch.zeros(size, dtype=torch.long)
        self._rewards = torch.zeros(size, dtype=torch.float32)
        self._valid_action_masks_next = torch.zeros(
            (size, 2 * MAX_SIZE_HAND + 2), dtype=torch.bool
        )
        self._game_over_flags = torch.zeros(size, dtype=torch.float32)

        # Current write index
        self._index: int = 0

        # Whether the buffer is full or not
        self._full: bool = False

        # RNG for sampling
        self._rng = np.random.default_rng(seed)

    @property
    def full(self) -> bool:
        return self._full

    @property
    def num_samples(self) -> int:
        if self._full:
            return self._size

        return self._index

    def store(self, sample: Sample) -> None:
        # Initialize tensors for state if not already initialized
        if self._states is None:
            state_shape = sample.state_t.shape

            self._states = torch.zeros((self._size, *state_shape), dtype=sample.state_t.dtype)
            self._states_next = torch.zeros(
                (self._size, *state_shape), dtype=sample.state_tp1.dtype
            )
            print(f"{self._states.shape=}")
            print(f"{self._states_next.shape=}")

        # Store the data in the appropriate location
        self._states[self._index] = sample.state_t
        self._states_next[self._index] = sample.state_tp1
        self._actions[self._index] = sample.action
        self._rewards[self._index] = sample.reward
        self._valid_action_masks_next[self._index] = sample.valid_action_mask_next
        self._game_over_flags[self._index] = sample.game_over_flag

        # Increment index
        self._index = (self._index + 1) % self._size

        # Set full if needed
        self._full = self._full or self._index == 0

    def sample(self, batch_size: int) -> Batch:
        if self._index < batch_size and not self._full:
            raise ValueError(f"Can't sample {batch_size} samples with only {self._index} samples")

        high = self._size if self._full else self._index
        sample_indexes = self._rng.integers(low=0, high=high, size=batch_size)

        # Gather sampled data
        return Batch(
            states=self._states[sample_indexes],
            states_next=self._states_next[sample_indexes],
            actions=self._actions[sample_indexes],
            rewards=self._rewards[sample_indexes],
            valid_action_masks_next=self._valid_action_masks_next[sample_indexes],
            game_over_flags=self._game_over_flags[sample_indexes],
        )
