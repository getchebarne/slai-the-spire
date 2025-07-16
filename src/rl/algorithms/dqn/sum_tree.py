class SumTree:
    def __init__(self, size: int):
        self._size = size

        self._nodes = [0.0] * (2 * size - 1)
        self._data = [0] * size
        self._num_entries = 0
        self._index = 0

    @property
    def total(self) -> float:
        return self._nodes[0]

    # update to the root node
    def _propagate(self, idx: int, delta: float) -> None:
        idx_parent = (idx - 1) // 2

        self._nodes[idx_parent] += delta

        if idx_parent != 0:
            self._propagate(idx_parent, delta)

    # find sample on leaf node
    def _retrieve(self, idx: int, cumsum: float) -> int:
        idx_left = 2 * idx + 1
        idx_right = idx_left + 1

        if idx_left >= len(self._nodes):
            return idx

        if cumsum <= self._nodes[idx_left]:
            return self._retrieve(idx_left, cumsum)

        return self._retrieve(idx_right, cumsum - self._nodes[idx_left])

    # store priority and sample
    def add(self, priority: float, data: int) -> None:
        idx_node = self._index + self._size - 1

        self._data[self._index] = data
        self.update(idx_node, priority)

        self._index = (self._index + 1) % self._size
        self._num_entries = min(self._size, self._num_entries + 1)

    # update priority
    def update(self, idx: int, priority: float) -> None:
        delta = priority - self._nodes[idx]

        self._nodes[idx] = priority
        self._propagate(idx, delta)

    # get priority and sample
    def get(self, cumsum: float) -> tuple[int, float, int]:
        idx_node = self._retrieve(0, cumsum)
        idx_data = idx_node - self._size + 1

        return idx_node, self._nodes[idx_node], self._data[idx_data]
