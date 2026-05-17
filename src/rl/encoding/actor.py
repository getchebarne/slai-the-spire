import slai


_MODIFIER_KIND_NAMES: list[str] = [m.name for m in slai.ModifierKind]
_MODIFIER_KIND_TO_IDX: dict[object, int] = {m: idx for idx, m in enumerate(slai.ModifierKind)}

_RL_NORMALIZATION_CAP = 30


def get_encoding_dim_actor_modifiers() -> int:
    return len(_MODIFIER_KIND_NAMES)


def encode_view_actor_modifiers(view_actor_modifiers: list[slai.Modifier]) -> list[float]:
    encoding = [0.0] * len(_MODIFIER_KIND_NAMES)
    for modifier in view_actor_modifiers:
        idx = _MODIFIER_KIND_TO_IDX.get(modifier.kind)
        if idx is None:
            continue
        stacks = abs(modifier.stacks)
        cap = min(modifier.stacks_max, _RL_NORMALIZATION_CAP)
        encoding[idx] = min(stacks, cap) / cap if cap > 0 else 0.0
    return encoding
