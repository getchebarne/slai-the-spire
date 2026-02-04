import numpy as np
import torch

from src.game.view.fsm import ViewFSM


_FSM_STATES = list(ViewFSM)
_FSM_DIM = len(_FSM_STATES)


def get_encoding_dim_fsm() -> int:
    return _FSM_DIM


def encode_batch_view_fsm(
    batch_view_fsm: list[ViewFSM], device: torch.device
) -> torch.Tensor:
    batch_size = len(batch_view_fsm)
    x_out = np.zeros((batch_size, _FSM_DIM), dtype=np.float32)

    for b, fsm in enumerate(batch_view_fsm):
        idx = _FSM_STATES.index(fsm)
        x_out[b, idx] = 1.0

    return torch.from_numpy(x_out).to(device)
