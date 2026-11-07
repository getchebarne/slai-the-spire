"""The training progress shared by curriculum updates and checkpoint I/O."""

from dataclasses import dataclass


__all__ = ["RunState"]


@dataclass
class RunState:
    """What a checkpoint resumes besides the weights, the optimizer and the iteration."""

    total_steps: int          # Steps played so far, every rank's games counted
    ascension_current: float  # The soft-curriculum position
    streak: int               # The consecutive passes at ascension_current so far
    lr_scale: float           # The Adaptive-KL learning rate scale
