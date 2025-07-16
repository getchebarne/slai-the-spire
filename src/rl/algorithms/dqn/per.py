from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from src.agents.dqn.sum_tree import SumTree

from src.game.combat.constant import MAX_MONSTERS
from src.game.combat.constant import MAX_SIZE_HAND


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


class PrioritizedReplayBuffer:
    def __init__(self, size: int, eps: float = 1e-2, alpha: float = 0.1, beta: float = 0.1):
        self._size = size
        self._eps = eps
        self._alpha = alpha
        self._beta = beta

        #
        self._max_priority = eps
        self._tree = SumTree(size)

        # Initially, these are None because we don't know the dimensions yet
        self._state_ts = None
        self._state_tp1s = None

        # Preallocate tensors with fixed shapes for scalars
        self._actions = torch.zeros(size, dtype=torch.long)
        self._rewards = torch.zeros(size, dtype=torch.float32)
        self._valid_action_mask_tp1s = torch.zeros(
            (size, MAX_SIZE_HAND + MAX_MONSTERS + 1), dtype=torch.float32
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
            print(f"{self._state_ts.shape=}")
            print(f"{self._state_tp1s.shape=}")

        #
        self._tree.add(self._max_priority, self._index)

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

        segment = self._tree.total / batch_size
        priorities = []
        idxs_data = []
        idxs_node = []

        for idx_batch in range(batch_size):
            cumsum = self._rng.uniform(segment * idx_batch, segment * (idx_batch + 1))
            idx_node, priority, idx_data = self._tree.get(cumsum)

            priorities.append(priority)
            idxs_data.append(idx_data)
            idxs_node.append(idx_node)

        priorities = np.array(priorities)
        is_weights = (self._tree._num_entries * (priorities / self._tree.total)) ** (
            1 / self._beta
        )
        is_weights /= is_weights.max()

        # Gather sampled data
        return (
            Batch(
                state_ts=self._state_ts[idxs_data],
                state_tp1s=self._state_tp1s[idxs_data],
                actions=self._actions[idxs_data],
                rewards=self._rewards[idxs_data],
                valid_action_mask_tp1s=self._valid_action_mask_tp1s[idxs_data],
                game_over_flags=self._game_over_flags[idxs_data],
            ),
            idxs_node,
            is_weights,
        )

    def update(self, idx: int, td_error: float) -> None:
        priority = (abs(td_error) + self._eps) ** self._alpha
        self._tree.update(idx, priority)
        self._max_priority = max(self._max_priority, priority)
