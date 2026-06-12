import numpy as np
from slai import Modifier
from slai import ModifierKind
from slai import members

from src.rl.utils import get_sqrt_norm


MODIFIER_KIND_TO_IDX = {modifier_kind: i for i, modifier_kind in enumerate(members(ModifierKind))}
# Sqrt-scaled cap; keeps resolution on small decision-relevant stacks while
# leaving headroom for Catalyst-doubled poison
STACKS_MAX = 99

# Buff/debuff split, mirroring the engine's MODIFIER_DEFS is_buff (slai modifier.rs).
# Hand-maintained because it isn't FFI-exposed; the assert below makes drift loud.
MODIFIER_IS_BUFF: dict[ModifierKind, bool] = {
    ModifierKind.Accuracy: True,
    ModifierKind.AfterImage: True,
    ModifierKind.Angry: True,
    ModifierKind.Artifact: True,
    ModifierKind.Asleep: True,
    ModifierKind.Blur: True,
    ModifierKind.Burst: True,
    ModifierKind.Choke: False,
    ModifierKind.CorpseExplosion: False,
    ModifierKind.CurlUp: True,
    ModifierKind.Dexterity: True,
    ModifierKind.DoubleDamage: True,
    ModifierKind.DrawCardNextTurn: True,
    ModifierKind.Enrage: True,
    ModifierKind.Entangled: False,
    ModifierKind.Envenom: True,
    ModifierKind.Frail: False,
    ModifierKind.InfiniteBlades: True,
    ModifierKind.Intangible: True,
    ModifierKind.Metallicize: True,
    ModifierKind.ModeShift: True,
    ModifierKind.NextTurnBlock: True,
    ModifierKind.NextTurnEnergy: True,
    ModifierKind.NoDraw: False,
    ModifierKind.NoxiousFumes: True,
    ModifierKind.Phantasmal: True,
    ModifierKind.PlatedArmor: True,
    ModifierKind.Poison: False,
    ModifierKind.Retain: True,
    ModifierKind.Ritual: True,
    ModifierKind.Shackled: False,
    ModifierKind.SharpHide: True,
    ModifierKind.Splittable: True,
    ModifierKind.SporeCloud: True,
    ModifierKind.Strength: True,
    ModifierKind.Thievery: True,
    ModifierKind.Thorns: True,
    ModifierKind.ThousandCuts: True,
    ModifierKind.ToolsOfTheTrade: True,
    ModifierKind.Vigor: True,
    ModifierKind.Vulnerable: False,
    ModifierKind.Weak: False,
    ModifierKind.WraithForm: False,
}
assert set(MODIFIER_IS_BUFF) == set(
    members(ModifierKind)
), "MODIFIER_IS_BUFF must cover every ModifierKind (sync with slai modifier.rs)"


def get_encoding_dim_modifiers() -> int:
    return len(MODIFIER_KIND_TO_IDX)


def encode_modifiers_into(modifiers: list[Modifier], pos: int, out: np.ndarray) -> int:
    for modifier in modifiers:
        idx = MODIFIER_KIND_TO_IDX[modifier.kind]
        # Boolean modifiers (engine ceiling 1) are presence flags; rest sqrt-scaled
        if modifier.stacks_max <= 1:
            out[pos + idx] = float(modifier.stacks)
        else:
            out[pos + idx] = get_sqrt_norm(modifier.stacks, STACKS_MAX)

    return pos + len(MODIFIER_KIND_TO_IDX)
