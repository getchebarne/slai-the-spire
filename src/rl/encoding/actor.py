"""Modifier-vector encoding shared by character + monster encoders.

slai exposes `ModifierKind` as a unit-enum class with the variants as class
attributes. We snapshot the variant list at module load and produce a fixed
dim vector; per-instance modifiers are accumulated by `kind` lookup.
"""

import slai


# Snapshot the runtime ModifierKind variants. The pyi may lag behind ffi.rs
# (43 variants in ffi.rs as of 2026-05). dir() reflects what's actually
# exposed, which is what env.step() will return.
_MODIFIER_KIND_NAMES: list[str] = sorted(
    n for n in dir(slai.ModifierKind) if not n.startswith("_")
)
_MODIFIER_KIND_TO_IDX: dict[object, int] = {
    getattr(slai.ModifierKind, name): idx for idx, name in enumerate(_MODIFIER_KIND_NAMES)
}

# Per-modifier stack normalization caps. Unknown / new modifiers default to
# DEFAULT_STACKS_MAX. Update opportunistically as the agent encounters new
# content; pinning these too low just compresses the activation, it won't
# crash.
DEFAULT_STACKS_MAX = 10
_STACKS_MAX_OVERRIDE: dict[str, int] = {
    "Strength": 20,
    "Weak": 5,
    "ModeShift": 60,
    "Ritual": 20,
    "SharpHide": 3,
    "SporeCloud": 2,
    "Vulnerable": 4,
    "Accuracy": 16,
    "NextTurnBlock": 20,
    "NextTurnEnergy": 5,
    "Blur": 5,
    "Dexterity": 12,
    "InfiniteBlades": 5,
    "AfterImage": 3,
    "Phantasmal": 2,
    "DoubleDamage": 1,
    "ThousandCuts": 4,
    "Burst": 4,
    "Poison": 30,
    "Frail": 5,
    "Metallicize": 20,
    "PlatedArmor": 20,
    "Vigor": 20,
}
_STACKS_MAX: list[float] = [
    float(_STACKS_MAX_OVERRIDE.get(name, DEFAULT_STACKS_MAX)) for name in _MODIFIER_KIND_NAMES
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
            continue  # unknown modifier kind (newer than this snapshot); skip
        stacks = abs(modifier.stacks)
        encoding[idx] = min(stacks, _STACKS_MAX[idx]) / _STACKS_MAX[idx]
    return encoding
