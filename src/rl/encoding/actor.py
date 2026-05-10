"""Modifier-vector encoding shared by character + monster encoders.

slai exposes `ModifierKind` as a unit-enum class with the variants as
class attributes, plus `Modifier.stacks_max_for(kind)` which returns the
engine's per-kind stack ceiling. We snapshot the variant list and stack
caps at module load.

slai's stack caps are often the soft "999" sentinel (effectively
unbounded for runtime purposes). For ML normalization that's too high —
values would all squash near zero. We additionally clamp to
`_RL_NORMALIZATION_CAP` (encoder concern, not engine concern).
"""

import slai


# Snapshot ModifierKind variants at module load. Iteration order is
# determined by the runtime `dir()`; sort to make the encoding stable.
_MODIFIER_KIND_NAMES: list[str] = sorted(
    n for n in dir(slai.ModifierKind) if not n.startswith("_")
)
_MODIFIER_KIND_TO_IDX: dict[object, int] = {
    getattr(slai.ModifierKind, name): idx for idx, name in enumerate(_MODIFIER_KIND_NAMES)
}

# Encoder-side normalization cap. slai's `stacks_max_for` returns 999 for
# many "effectively unbounded" modifiers (Strength, Burst, etc.); a tighter
# RL-side cap keeps activations in a reasonable range.
_RL_NORMALIZATION_CAP = 30
_STACKS_MAX: list[float] = [
    float(min(slai.Modifier.stacks_max_for(getattr(slai.ModifierKind, name)), _RL_NORMALIZATION_CAP))
    for name in _MODIFIER_KIND_NAMES
]


def get_encoding_dim_actor_modifiers() -> int:
    """Return the dimension of actor modifier encoding."""
    return len(_MODIFIER_KIND_NAMES)


def encode_view_actor_modifiers(view_actor_modifiers: list[slai.Modifier]) -> list[float]:
    """Encode a list of `slai.Modifier` into a fixed-length normalized vector."""
    encoding = [0.0] * len(_MODIFIER_KIND_NAMES)
    for modifier in view_actor_modifiers:
        idx = _MODIFIER_KIND_TO_IDX.get(modifier.kind)
        if idx is None:
            continue  # newer modifier kind than this snapshot; skip
        # Stacks can be negative for some modifiers (Strength under Decay,
        # Wraith Form interactions) — encode magnitude.
        stacks = abs(modifier.stacks)
        cap = _STACKS_MAX[idx]
        encoding[idx] = min(stacks, cap) / cap if cap > 0 else 0.0
    return encoding
