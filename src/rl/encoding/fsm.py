import numpy as np
import torch

from src.game.view.fsm import ViewFSM


_FSM_STATES = list(ViewFSM)
FSM_DIM = len(_FSM_STATES)


def encode_batch_view_fsm(batch_view_fsm: list[ViewFSM], device: torch.device) -> torch.Tensor:
    batch_size = len(batch_view_fsm)
    x_out = np.zeros((batch_size, FSM_DIM), dtype=np.float32)

    for b, fsm in enumerate(batch_view_fsm):
        idx = _FSM_STATES.index(fsm)
        x_out[b, idx] = 1.0

    return torch.from_numpy(x_out).to(device)
