"""One-hot encoder for the engine's phase.

Each `slai.Phase.*` variant subclass occupies one slot. Order is stable;
reordering invalidates trained models keyed by the resulting one-hot.
"""

import numpy as np
import slai
import torch


# 9 phase slots — one per slai.Phase variant.
_PHASE_CLASSES: list[type] = [
    slai.Phase.Map,
    slai.Phase.CombatDefault,
    slai.Phase.CombatAwaitDiscard,
    slai.Phase.CombatAwaitRetain,
    slai.Phase.CombatAwaitNightmare,
    slai.Phase.CombatAwaitSetup,
    slai.Phase.CombatReward,
    slai.Phase.RestSite,
    slai.Phase.GameOver,
]
FSM_DIM = len(_PHASE_CLASSES)


def encode_batch_view_fsm(batch_view_phase: list[object], device: torch.device) -> torch.Tensor:
    batch_size = len(batch_view_phase)
    x_out = np.zeros((batch_size, FSM_DIM), dtype=np.float32)

    for b, phase in enumerate(batch_view_phase):
        for idx, phase_cls in enumerate(_PHASE_CLASSES):
            if isinstance(phase, phase_cls):
                x_out[b, idx] = 1.0
                break

    return torch.from_numpy(x_out).to(device)
