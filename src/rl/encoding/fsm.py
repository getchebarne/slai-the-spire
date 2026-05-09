"""One-hot encoder for the engine's phase (was FSM in the in-tree emulator).

slai's `Phase` is an "open" enum: `slai.Phase.<Variant>` are real classes
that the runtime returns instances of. Use `isinstance` to dispatch. Phase
variants beyond the listed ones (e.g. `CombatAwaitNightmare`,
`CombatAwaitRetain`, `CombatAwaitSetup`) collapse to the all-zero "unknown"
slot — they're out of scope for the current trainer.
"""

import numpy as np
import slai
import torch


# Phases the trainer can route. Order is stable; index into `_PHASE_CLASSES`
# is the one-hot slot. Add a new entry only after wiring the new phase
# through route.py / masks.py / heads.
_PHASE_CLASSES: list[type] = [
    slai.Phase.Map,
    slai.Phase.CombatDefault,
    slai.Phase.CombatAwaitDiscard,
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
        # else: unknown phase (Nightmare/Retain/Setup) → all-zero slot

    return torch.from_numpy(x_out).to(device)
