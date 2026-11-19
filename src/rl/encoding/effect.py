"""Per-effect feature rows.

Each `Effect` becomes one factored row — Effect kind OHE + target features + typed
param channels + magnitudes on shared physical scales — embedded model-side by
one shared MLP and added into its owning Entity's token, scaled by 1/sqrt(the
owner's count of Effect rows). A pending halt's Effect row has no owning token:
it is global context.

Two rules the layout depends on:

- **Referents contribute identity, never Effects.** `CardAdd{Bite}` writes a
  name-vocabulary column, not Bite's own encoding. That makes the row depth-0 by
  construction, which matters because the engine's Effect-reference graph
  contains a self-loop (Endless Agony adds itself).
- **Normalization is scoped** — see `_normalize_effect_row`.
"""

import math
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from enum import StrEnum
from enum import auto
from types import UnionType
from typing import Any, Protocol, SupportsInt, get_args

import numpy as np
import slai
from slai import Amount
from slai import AmountAbsolute
from slai import AmountRange
from slai import AmountRelative
from slai import CandidateFilter
from slai import CandidatePool
from slai import CardColor
from slai import CardKind
from slai import CardPile
from slai import CardRarity
from slai import CostScope
from slai import DeltaSign
from slai import Effect
from slai import ModifierKind
from slai import ReadAt
from slai import RelicExclusion
from slai import RelicTier
from slai import RewardRollTrigger
from slai import Rounding
from slai import SelectionKind
from slai import get_card_templates
from slai import get_event_option_templates
from slai import get_monster_templates
from slai import get_potion_templates
from slai import get_relic_templates

from src.rl.constants import ASCENSION_LEVEL
from src.rl.constants import MAX_SIZE_CARD_PILE_DECK
from src.rl.constants import MAX_SIZE_CARD_PILE_HAND
from src.rl.constants import MULTI_STAB_INSTANCES_MAX
from src.rl.encoding.layout import make_channel_allocator
from src.rl.types import EffectSource


__all__ = [
    "ENCODING_DIM_EFFECT",
    "ENCODING_DIM_SPARSE_EFFECT",
    "catalog_effects",
    "make_effect_row",
]


# Live Card / Relic amounts arrive Modifier-scaled, so a catalog maximum is a
# floor rather than a ceiling; past this the channel clamps
_LIVE_SCALE_HEADROOM = 1.5
_ASCENSIONS = range(ASCENSION_LEVEL + 1)


def catalog_effects() -> Iterator[Effect]:
    """Yield every Effect reachable from a static definition, at every ascension.

    Yields
    ------
    Effect
        One Effect of a Card, Potion or Relic template, or of an Event Option or a Monster move
        at an ascension from 0 to ASCENSION_LEVEL.
    """

    # Cards
    for template in get_card_templates():
        yield from (*template.effects_play, *template.effects_discard, *template.effects_draw)

    # Potions
    for potion in get_potion_templates():
        yield from potion.effects

    # Relics, one sweep per trigger field
    for relic in get_relic_templates():
        for effects in (
            relic.effects_combat_start,
            relic.effects_turn_start,
            relic.effects_turn_end,
            relic.effects_combat_end,
            relic.effects_pickup,
            relic.effects_rest,
            relic.effects_counter,
        ):
            yield from effects

    # Event Options
    for ascension in _ASCENSIONS:
        for option in get_event_option_templates(ascension):
            yield from option.effects

    # Monster moves
    for ascension in _ASCENSIONS:
        for monster in get_monster_templates(ascension):
            for move in monster.moves:
                yield from move.effects


# Bucketed by exact type once
_BY_TYPE: dict = {}
for _effect in catalog_effects():
    _BY_TYPE.setdefault(type(_effect), []).append(_effect)
del _effect


def _amount_parts(amount: Amount) -> tuple[float, float, bool]:
    """Split an Amount union into (absolute, relative_fraction, is_range).

    A range folds to its midpoint in the absolute channel.

    Parameters
    ----------
    amount : Amount
        The amount to split.

    Returns
    -------
    absolute : float
        An absolute amount, or a range's midpoint; 0.0 for a relative one.
    relative_fraction : float
        A relative amount's numerator / denominator; 0.0 otherwise.
    is_range : bool
        Whether the amount is a range.
    """
    if isinstance(amount, AmountAbsolute):
        return float(amount.amount), 0.0, False
    if isinstance(amount, AmountRelative):
        return 0.0, amount.numerator / amount.denominator, False
    if isinstance(amount, AmountRange):
        return (amount.min + amount.max) / 2, 0.0, True

    raise ValueError(f"Unsupported Amount variant: {type(amount).__name__}")


class _MirrorEnum(SupportsInt, Protocol):
    @staticmethod
    def members() -> "list[Any]":
        ...


def _field[E, V](cls: type[E], get: Callable[[E], V]) -> Iterator[V]:
    """Yield one field across every catalog instance of an Effect type.

    Parameters
    ----------
    cls : type[E]
        The Effect type.
    get : Callable[[E], V]
        The field's getter, called on each instance.

    Yields
    ------
    V
        The field of one catalog instance of `cls`.
    """
    effects: list[E] = _BY_TYPE.get(cls, [])
    for effect in effects:
        yield get(effect)


def _cap(*specs: Iterable[Amount | float], floor: float = 1.0) -> float:
    """Return the max |value| over catalog fields sharing a physical scale.

    Parameters
    ----------
    *specs : Iterable[Amount | float]
        Each field's values; an Amount counts as its absolute part, a range as its midpoint.
    floor : float, default 1.0
        The smallest cap returned.

    Returns
    -------
    float
        The largest |value|, at least `floor`.
    """
    max_ = floor
    for spec in specs:
        for value in spec:
            if isinstance(value, (AmountAbsolute, AmountRelative, AmountRange)):
                value_scalar = _amount_parts(value)[0]
            else:
                value_scalar = float(value)

            max_ = max(max_, abs(value_scalar))

    return max_


def _vocabulary[M: _MirrorEnum](*specs: Iterable[M | list[M]]) -> dict[M, int]:
    """Build a catalog-derived identity vocabulary; index 0 is reserved as unseen.

    Parameters
    ----------
    *specs : Iterable[M | list[M]]
        Each field's values; a list adds each of its names.

    Returns
    -------
    dict[M, int]
        Each name, mapped to its index from 1, in the names' int order.
    """
    names: set[M] = set()
    for spec in specs:
        for value in spec:
            names.update(value if isinstance(value, list) else [value])

    return {name: idx_name + 1 for idx_name, name in enumerate(sorted(names, key=int))}


def _index_enum[M: _MirrorEnum](enum: type[M]) -> dict[M, int]:
    """Index the engine enum's members by their integer discriminants.

    Parameters
    ----------
    enum : type[M]
        The engine enum, exposing its members in declaration order.

    Returns
    -------
    dict[M, int]
        Each member's column index.
    """
    return {member: int(member) for member in enum.members()}


def _index_union(union: UnionType) -> dict[type, int]:
    """Index the variants of an engine union in declaration order.

    Parameters
    ----------
    union : UnionType
        The engine union whose variants share a categorical channel.

    Returns
    -------
    dict[type, int]
        Each variant type's column index.
    """
    return {cls: idx_class for idx_class, cls in enumerate(get_args(union))}


_INDEX_EFFECT = _index_union(Effect)
_INDEX_CAND_POOL = _index_enum(CandidatePool)
_INDEX_SELECTION_KIND = _index_union(SelectionKind)
_INDEX_CAND_FILTER = _index_enum(CandidateFilter)
_INDEX_MODIFIER_KIND = _index_enum(ModifierKind)
_INDEX_REWARD_ROLL_TRIGGER = _index_enum(RewardRollTrigger)
_INDEX_DELTA_SIGN = _index_enum(DeltaSign)
_INDEX_CARD_PILE = _index_enum(CardPile)
_INDEX_CARD_KIND = _index_enum(CardKind)
_INDEX_CARD_COLOR = _index_enum(CardColor)
_INDEX_CARD_RARITY = _index_enum(CardRarity)
_INDEX_RELIC_TIER = _index_enum(RelicTier)
_INDEX_RELIC_EXCLUSION = _index_enum(RelicExclusion)
_INDEX_COST_SCOPE = _index_enum(CostScope)
_INDEX_READ_AT = _index_enum(ReadAt)
_INDEX_ROUNDING = _index_enum(Rounding)

_VOCAB_CARD = _vocabulary(_field(slai.EffectCardAdd, lambda effect: effect.card_name))
_VOCAB_RELIC = _vocabulary(
    _field(slai.EffectRelicGrantSpecific, lambda effect: effect.name),
    _field(slai.EffectRelicGrantPool, lambda effect: effect.pool),
    _field(slai.EffectRewardRollRelic, lambda effect: effect.pool),
)
_VOCAB_MONSTER = _vocabulary(
    _field(slai.EffectMonsterSpawn, lambda effect: effect.name),
    _field(slai.EffectMonsterSplit, lambda effect: effect.name),
    _field(slai.EffectMonsterSummon, lambda effect: effect.pool),
)


# Caps
_CAP_DAMAGE = _LIVE_SCALE_HEADROOM * _cap(
    _field(slai.EffectDamagePhysical, lambda effect: effect.amount),
    _field(slai.EffectHeelHookProc, lambda effect: effect.damage),
    _field(slai.EffectSkewerProc, lambda effect: effect.damage),
    _field(slai.EffectDamageFinisher, lambda effect: effect.damage),
    _field(slai.EffectDamageFlechettes, lambda effect: effect.damage),
    _field(slai.EffectDamageMindBlast, lambda effect: effect.damage),
    _field(slai.EffectBombArm, lambda effect: effect.damage),
    _field(slai.EffectRitualDaggerProc, lambda effect: effect.bump),
    _field(slai.EffectGlassKnifeDecay, lambda effect: effect.delta),
)

# The catalog's Multi-Stab is a 0-hit placeholder, so its live hit count is bridged
_CAP_INSTANCES = max(
    _cap(_field(slai.EffectDamagePhysical, lambda effect: effect.instances)),
    float(MULTI_STAB_INSTANCES_MAX),
)
_CAP_BLOCK = _LIVE_SCALE_HEADROOM * _cap(
    _field(slai.EffectBlockGain, lambda effect: effect.amount),
    _field(slai.EffectEscapePlanCheck, lambda effect: effect.block),
)
_CAP_HEALTH = _cap(
    _field(slai.EffectHealthDelta, lambda effect: effect.amount),
    _field(slai.EffectHealthSet, lambda effect: effect.amount),
    _field(slai.EffectMaxHealthDelta, lambda effect: effect.amount),
)
_CAP_GOLD = _cap(
    _field(slai.EffectGoldDelta, lambda effect: effect.amount),
    _field(slai.EffectHandOfGreedProc, lambda effect: effect.gold),
    _field(slai.EffectRewardRollGold, lambda effect: effect.amount),
)
_CAP_ENERGY = _cap(
    _field(slai.EffectEnergyDelta, lambda effect: effect.amount),
    _field(slai.EffectSneakyStrikeProc, lambda effect: effect.energy),
)
_CAP_STACKS = _cap(
    _field(slai.EffectModifierDelta, lambda effect: effect.stacks),
    _field(slai.EffectStrengthLoseTemp, lambda effect: effect.stacks),
)
_CAP_COUNT = max(
    _cap(
        _field(slai.EffectCardAdd, lambda effect: effect.count),
        _field(slai.EffectCardAddRandom, lambda effect: effect.count),
        _field(slai.EffectCardDiscoverSelect, lambda effect: effect.copies),
        _field(slai.EffectCardDraw, lambda effect: effect.count),
        _field(slai.EffectCardDrawIfNoAttacks, lambda effect: effect.count),
        _field(slai.EffectCardDrawUpTo, lambda effect: effect.amount),
        _field(slai.EffectCardDiscoverRoll, lambda effect: effect.count),
        _field(slai.EffectRewardRollPotions, lambda effect: effect.count),
    ),
    float(MAX_SIZE_CARD_PILE_HAND),
)
_CAP_DELTA = _cap(_field(slai.EffectEventAdvanceState, lambda effect: effect.delta))
_CAP_FACTOR = _cap(_field(slai.EffectModifierMultiply, lambda effect: effect.factor))
_CAP_COST = _cap(_field(slai.EffectSetCostOverride, lambda effect: effect.amount))
_CAP_TURNS = float(slai.GameEnv.BOMB_FUSE_TURNS)
_CAP_SELECT_COUNT = float(MAX_SIZE_CARD_PILE_DECK)
_CAP_RELIC_POOL_SIZE = float(max(len(_VOCAB_RELIC), 1))


# Contiguous groups; every column carries a cap and a mode


def _bucket_width(cap: float) -> int:
    """Return the number of square-root buckets through the channel's cap.

    Parameters
    ----------
    cap : float
        The channel's nonnegative magnitude cap.

    Returns
    -------
    int
        One column per integer bucket, including zero.
    """
    return int(math.sqrt(cap)) + 1


_CHANNEL_LAYOUT_EFFECT, _alloc = make_channel_allocator()


# Effect's kind: one-hot over the Effect union
_OFF_EFFECT_KIND = _alloc("effect_kind", len(_INDEX_EFFECT))

# Target
_OFF_TARGETED = _alloc("targeted")
_OFF_CANDIDATE_POOL = _alloc("candidate_pool", len(_INDEX_CAND_POOL))
_OFF_SELECTION_KIND = _alloc("selection_kind", len(_INDEX_SELECTION_KIND))
_OFF_SELECT_COUNT = _alloc("select_count")
_OFF_CANDIDATE_FILTER = _alloc("candidate_filter", len(_INDEX_CAND_FILTER))
_OFF_EFFECT_SOURCE = _alloc("effect_source", len(EffectSource))  # Written by make_effect_row

# Categorical params, named for the engine type they one-hot
_OFF_MODIFIER_KIND = _alloc("modifier_kind", len(_INDEX_MODIFIER_KIND))
_OFF_MODIFIER_BUFF = _alloc("modifier_buff")
_OFF_DELTA_SIGN = _alloc("delta_sign", len(_INDEX_DELTA_SIGN))
_OFF_CARD_PILE = _alloc("card_pile", len(_INDEX_CARD_PILE))
_OFF_CARD_KIND = _alloc("card_kind", len(_INDEX_CARD_KIND))
_OFF_CARD_COLOR = _alloc("card_color", len(_INDEX_CARD_COLOR))
_OFF_CARD_RARITY = _alloc("card_rarity", len(_INDEX_CARD_RARITY))
_OFF_RELIC_TIER = _alloc("relic_tier", len(_INDEX_RELIC_TIER))
_OFF_RELIC_EXCLUSION = _alloc("relic_exclusion", len(_INDEX_RELIC_EXCLUSION))
_OFF_COST_SCOPE = _alloc("cost_scope", len(_INDEX_COST_SCOPE))
_OFF_COST_ZERO_PRESENT = _alloc("cost_zero_present")
_OFF_REWARD_ROLL_TRIGGER = _alloc("reward_roll_trigger", len(_INDEX_REWARD_ROLL_TRIGGER))
_OFF_READ_AT = _alloc("read_at", len(_INDEX_READ_AT))
_OFF_ROUNDING = _alloc("rounding", len(_INDEX_ROUNDING))

# Identity vocabularies, named for the engine name enum (index 0 = unseen)
_OFF_CARD_NAME = _alloc("card_name", len(_VOCAB_CARD) + 1)
_OFF_RELIC_NAME = _alloc("relic_name", len(_VOCAB_RELIC) + 1)
_OFF_MONSTER_NAME = _alloc("monster_name", len(_VOCAB_MONSTER) + 1)
_OFF_RELIC_POOL_SIZE = _alloc("relic_pool_size")

# Magnitudes, one per physical unit
_OFF_DAMAGE = _alloc("damage")
_OFF_DAMAGE_BUCKETS = _alloc("damage_buckets", _bucket_width(_CAP_DAMAGE))
_OFF_INSTANCES = _alloc("instances")
_OFF_BLOCK = _alloc("block")
_OFF_HEALTH_ABSOLUTE = _alloc("health_absolute")
_OFF_HEALTH_BUCKETS = _alloc("health_buckets", _bucket_width(_CAP_HEALTH))
_OFF_HEALTH_RELATIVE = _alloc("health_relative")
_OFF_GOLD_ABSOLUTE = _alloc("gold_absolute")
_OFF_GOLD_BUCKETS = _alloc("gold_buckets", _bucket_width(_CAP_GOLD))
_OFF_GOLD_RELATIVE = _alloc("gold_relative")
_OFF_AMOUNT_IS_RANGE = _alloc("amount_is_range")
_OFF_ENERGY = _alloc("energy")
_OFF_STACKS = _alloc("stacks")
_OFF_STACKS_NEGATIVE = _alloc("stacks_negative")
_OFF_STACKS_BUCKETS = _alloc("stacks_buckets", _bucket_width(_CAP_STACKS))
_OFF_COUNT = _alloc("count")
_OFF_DELTA = _alloc("delta")
_OFF_CHANCE = _alloc("chance")
_OFF_FACTOR = _alloc("factor")
_OFF_COST = _alloc("cost")
_OFF_TURNS = _alloc("turns")

# Boolean flags
# Boolean flags, named for the engine field
_OFF_ADVANCE_ON_MISS = _alloc("advance_on_miss")
_OFF_BLOCK_FROM_HIT = _alloc("block_from_hit")
_OFF_BOTTOM = _alloc("bottom")
_OFF_CHOOSE_DISCARDS = _alloc("choose_discards")
_OFF_COLORLESS = _alloc("colorless")
_OFF_ESCAPED_CHARACTER = _alloc("escaped_character")
_OFF_FREE = _alloc("free")
_OFF_IF_POISONED = _alloc("if_poisoned")
_OFF_LIFESTEAL = _alloc("lifesteal")
_OFF_LIMITED = _alloc("limited")
_OFF_ON_OWNER = _alloc("on_owner")
_OFF_ONLY_REDUCE = _alloc("only_reduce")
_OFF_RANDOM = _alloc("random")
_OFF_RARE_ONLY = _alloc("rare_only")
_OFF_UNIFORM = _alloc("uniform")
_OFF_UPGRADED = _alloc("upgraded")

# (float channel, bucket block, cap) — buckets derive from the final raw floats
_BUCKET_META: tuple[tuple[int, int, float], ...] = (
    (_OFF_DAMAGE, _OFF_DAMAGE_BUCKETS, _CAP_DAMAGE),
    (_OFF_HEALTH_ABSOLUTE, _OFF_HEALTH_BUCKETS, _CAP_HEALTH),
    (_OFF_GOLD_ABSOLUTE, _OFF_GOLD_BUCKETS, _CAP_GOLD),
    (_OFF_STACKS, _OFF_STACKS_BUCKETS, _CAP_STACKS),
)

# Total
ENCODING_DIM_EFFECT = sum(w for _, w in _CHANNEL_LAYOUT_EFFECT)

# The channels that set several of a row's columns
_MULTI_HOT_CHANNELS = (
    "candidate_filter",
    "relic_name",
    "monster_name",
)

# Most nonzero columns an Effect row holds: one per one-hot or scalar channel, and every column
# of the multi-hot ones
ENCODING_DIM_SPARSE_EFFECT = sum(
    width if channel in _MULTI_HOT_CHANNELS else 1 for channel, width in _CHANNEL_LAYOUT_EFFECT
)


class _NormMode(StrEnum):
    SQRT = auto()
    LINEAR = auto()


# (offset, cap, mode) for every non-unit channel
_NORM_META: tuple[tuple[int, float, _NormMode], ...] = (
    (_OFF_SELECT_COUNT, _CAP_SELECT_COUNT, _NormMode.LINEAR),
    (_OFF_RELIC_POOL_SIZE, _CAP_RELIC_POOL_SIZE, _NormMode.LINEAR),
    (_OFF_DAMAGE, _CAP_DAMAGE, _NormMode.SQRT),
    (_OFF_INSTANCES, _CAP_INSTANCES, _NormMode.LINEAR),
    (_OFF_BLOCK, _CAP_BLOCK, _NormMode.LINEAR),
    (_OFF_HEALTH_ABSOLUTE, _CAP_HEALTH, _NormMode.SQRT),
    (_OFF_GOLD_ABSOLUTE, _CAP_GOLD, _NormMode.SQRT),
    (_OFF_ENERGY, _CAP_ENERGY, _NormMode.LINEAR),
    (_OFF_STACKS, _CAP_STACKS, _NormMode.SQRT),
    (_OFF_COUNT, _CAP_COUNT, _NormMode.LINEAR),
    (_OFF_DELTA, _CAP_DELTA, _NormMode.LINEAR),
    (_OFF_CHANCE, 100.0, _NormMode.LINEAR),
    (_OFF_FACTOR, _CAP_FACTOR, _NormMode.LINEAR),
    (_OFF_COST, _CAP_COST, _NormMode.LINEAR),
    (_OFF_TURNS, _CAP_TURNS, _NormMode.LINEAR),
)

_NP_SCALED_MASK = np.zeros(ENCODING_DIM_EFFECT, dtype=bool)
_NP_CAP = np.ones(ENCODING_DIM_EFFECT, dtype=np.float32)
_NP_SQRT_MASK = np.zeros(ENCODING_DIM_EFFECT, dtype=bool)
for _off, _cap_value, _mode in _NORM_META:
    _NP_SCALED_MASK[_off] = True
    _NP_CAP[_off] = _cap_value
    _NP_SQRT_MASK[_off] = _mode == _NormMode.SQRT

# Only the _NORM_META channels need scaling at all
_NP_IDX_SCALED = np.nonzero(_NP_SCALED_MASK)[0]
_NP_SCALED_CAP = _NP_CAP[_NP_IDX_SCALED]
_NP_SCALED_SQRT = _NP_SQRT_MASK[_NP_IDX_SCALED]
_NP_SCALED_LINEAR = ~_NP_SCALED_SQRT


# Per-effect-kind encoders
def _write_amount(np_out: np.ndarray, amount: Amount, off_abs: int, off_rel: int) -> None:
    """Write an Amount union into absolute/relative channels, plus the shared range flag.

    Parameters
    ----------
    np_out : np.ndarray
        (ENCODING_DIM_EFFECT,) float32. The Effect's raw row, updated in place.
    amount : Amount
        The amount; a relative one also writes its ReadAt and Rounding one-hots.
    off_abs : int
        The absolute channel's column.
    off_rel : int
        The relative channel's column.
    """
    absolute, relative, is_range = _amount_parts(amount)
    np_out[off_abs] += absolute
    np_out[off_rel] += relative
    if is_range:
        np_out[_OFF_AMOUNT_IS_RANGE] = 1.0

    # A fraction also says which value it reads and how it rounds
    if isinstance(amount, AmountRelative):
        np_out[_OFF_READ_AT + _INDEX_READ_AT[amount.read_at]] = 1.0
        np_out[_OFF_ROUNDING + _INDEX_ROUNDING[amount.rounding]] = 1.0


def _cost_zero(np_out: np.ndarray, value: CostScope | None) -> None:
    """Write an Optional[CostScope]: a presence flag plus the scope one-hot.

    Parameters
    ----------
    np_out : np.ndarray
        (ENCODING_DIM_EFFECT,) float32. The Effect's raw row, updated in place.
    value : CostScope | None
        The Effect's `cost_zero` scope; None writes nothing.
    """
    if value is not None:
        np_out[_OFF_COST_ZERO_PRESENT] = 1.0
        np_out[_OFF_COST_SCOPE + _INDEX_COST_SCOPE[value]] = 1.0


def _encode_block_gain(np_out: np.ndarray, effect: slai.EffectBlockGain) -> None:
    np_out[_OFF_BLOCK] += effect.amount


def _encode_bomb_arm(np_out: np.ndarray, effect: slai.EffectBombArm) -> None:
    np_out[_OFF_DAMAGE] += effect.damage
    np_out[_OFF_TURNS] += effect.turns


def _encode_card_add(np_out: np.ndarray, effect: slai.EffectCardAdd) -> None:
    np_out[_OFF_CARD_NAME + _VOCAB_CARD.get(effect.card_name, 0)] = 1.0
    np_out[_OFF_COUNT] += effect.count
    np_out[_OFF_CARD_PILE + _INDEX_CARD_PILE[effect.card_pile]] = 1.0
    if effect.upgraded:
        np_out[_OFF_UPGRADED] = 1.0


def _encode_card_add_random(np_out: np.ndarray, effect: slai.EffectCardAddRandom) -> None:
    np_out[_OFF_CARD_COLOR + _INDEX_CARD_COLOR[effect.color]] = 1.0
    _cost_zero(np_out, effect.cost_zero)
    np_out[_OFF_COUNT] += effect.count
    if effect.kind is not None:
        np_out[_OFF_CARD_KIND + _INDEX_CARD_KIND[effect.kind]] = 1.0
    np_out[_OFF_CARD_PILE + _INDEX_CARD_PILE[effect.card_pile]] = 1.0
    if effect.rarity is not None:
        np_out[_OFF_CARD_RARITY + _INDEX_CARD_RARITY[effect.rarity]] = 1.0
    if effect.upgraded:
        np_out[_OFF_UPGRADED] = 1.0


def _encode_card_discover_select(
    np_out: np.ndarray, effect: slai.EffectCardDiscoverSelect
) -> None:
    np_out[_OFF_COUNT] += effect.copies
    _cost_zero(np_out, effect.cost_zero)
    np_out[_OFF_CARD_PILE + _INDEX_CARD_PILE[effect.card_pile]] = 1.0


def _encode_card_discover_roll(np_out: np.ndarray, effect: slai.EffectCardDiscoverRoll) -> None:
    np_out[_OFF_CARD_COLOR + _INDEX_CARD_COLOR[effect.color]] = 1.0
    np_out[_OFF_COUNT] += effect.count
    if effect.kind is not None:
        np_out[_OFF_CARD_KIND + _INDEX_CARD_KIND[effect.kind]] = 1.0


def _encode_card_draw(np_out: np.ndarray, effect: slai.EffectCardDraw) -> None:
    np_out[_OFF_COUNT] += effect.count


def _encode_card_draw_if_no_attacks(
    np_out: np.ndarray, effect: slai.EffectCardDrawIfNoAttacks
) -> None:
    np_out[_OFF_COUNT] += effect.count


def _encode_card_draw_up_to(np_out: np.ndarray, effect: slai.EffectCardDrawUpTo) -> None:
    np_out[_OFF_COUNT] += effect.amount


def _encode_card_duplicate(np_out: np.ndarray, effect: slai.EffectCardDuplicate) -> None:
    np_out[_OFF_CARD_PILE + _INDEX_CARD_PILE[effect.card_pile]] = 1.0


def _encode_card_move(np_out: np.ndarray, effect: slai.EffectCardMove) -> None:
    _cost_zero(np_out, effect.cost_zero)
    np_out[_OFF_CARD_PILE + _INDEX_CARD_PILE[effect.card_pile]] = 1.0


def _encode_card_setup_select(np_out: np.ndarray, effect: slai.EffectCardSetupSelect) -> None:
    if effect.bottom:
        np_out[_OFF_BOTTOM] = 1.0
    if effect.free:
        np_out[_OFF_FREE] = 1.0


def _encode_card_transform(np_out: np.ndarray, effect: slai.EffectCardTransform) -> None:
    if effect.upgraded:
        np_out[_OFF_UPGRADED] = 1.0


def _encode_combat_end(np_out: np.ndarray, effect: slai.EffectCombatEnd) -> None:
    if effect.escaped_character:
        np_out[_OFF_ESCAPED_CHARACTER] = 1.0


def _encode_damage_finisher(np_out: np.ndarray, effect: slai.EffectDamageFinisher) -> None:
    np_out[_OFF_DAMAGE] += effect.damage


def _encode_damage_flechettes(np_out: np.ndarray, effect: slai.EffectDamageFlechettes) -> None:
    np_out[_OFF_DAMAGE] += effect.damage


def _encode_damage_mind_blast(np_out: np.ndarray, effect: slai.EffectDamageMindBlast) -> None:
    np_out[_OFF_DAMAGE] += effect.damage


def _encode_damage_physical(np_out: np.ndarray, effect: slai.EffectDamagePhysical) -> None:
    np_out[_OFF_DAMAGE] += effect.amount
    if effect.block_from_hit:
        np_out[_OFF_BLOCK_FROM_HIT] = 1.0
    if effect.if_poisoned:
        np_out[_OFF_IF_POISONED] = 1.0
    np_out[_OFF_INSTANCES] += effect.instances
    if effect.lifesteal:
        np_out[_OFF_LIFESTEAL] = 1.0


def _encode_damage_deal(np_out: np.ndarray, effect: slai.EffectDamageDeal) -> None:
    np_out[_OFF_DAMAGE] += effect.amount
    if effect.lifesteal:
        np_out[_OFF_LIFESTEAL] = 1.0


def _encode_energy_delta(np_out: np.ndarray, effect: slai.EffectEnergyDelta) -> None:
    np_out[_OFF_ENERGY] += effect.amount
    np_out[_OFF_DELTA_SIGN + _INDEX_DELTA_SIGN[effect.sign]] = 1.0


def _encode_escape_plan_check(np_out: np.ndarray, effect: slai.EffectEscapePlanCheck) -> None:
    np_out[_OFF_BLOCK] += effect.block


def _encode_event_advance_state(np_out: np.ndarray, effect: slai.EffectEventAdvanceState) -> None:
    np_out[_OFF_DELTA] += effect.delta


def _encode_gamble(np_out: np.ndarray, effect: slai.EffectGamble) -> None:
    if effect.choose_discards:
        np_out[_OFF_CHOOSE_DISCARDS] = 1.0


def _encode_glass_knife_decay(np_out: np.ndarray, effect: slai.EffectGlassKnifeDecay) -> None:
    np_out[_OFF_DAMAGE] += effect.delta


def _encode_gold_delta(np_out: np.ndarray, effect: slai.EffectGoldDelta) -> None:
    _write_amount(np_out, effect.amount, _OFF_GOLD_ABSOLUTE, _OFF_GOLD_RELATIVE)
    np_out[_OFF_DELTA_SIGN + _INDEX_DELTA_SIGN[effect.sign]] = 1.0


def _encode_hand_of_greed_proc(np_out: np.ndarray, effect: slai.EffectHandOfGreedProc) -> None:
    np_out[_OFF_GOLD_ABSOLUTE] += effect.gold


def _encode_health_delta(np_out: np.ndarray, effect: slai.EffectHealthDelta) -> None:
    _write_amount(np_out, effect.amount, _OFF_HEALTH_ABSOLUTE, _OFF_HEALTH_RELATIVE)
    np_out[_OFF_DELTA_SIGN + _INDEX_DELTA_SIGN[effect.sign]] = 1.0


def _encode_health_set(np_out: np.ndarray, effect: slai.EffectHealthSet) -> None:
    _write_amount(np_out, effect.amount, _OFF_HEALTH_ABSOLUTE, _OFF_HEALTH_RELATIVE)


def _encode_heel_hook_proc(np_out: np.ndarray, effect: slai.EffectHeelHookProc) -> None:
    np_out[_OFF_DAMAGE] += effect.damage


def _encode_joust_bet(np_out: np.ndarray, effect: slai.EffectJoustBet) -> None:
    if effect.on_owner:
        np_out[_OFF_ON_OWNER] = 1.0


def _encode_max_health_delta(np_out: np.ndarray, effect: slai.EffectMaxHealthDelta) -> None:
    _write_amount(np_out, effect.amount, _OFF_HEALTH_ABSOLUTE, _OFF_HEALTH_RELATIVE)
    np_out[_OFF_DELTA_SIGN + _INDEX_DELTA_SIGN[effect.sign]] = 1.0


def _encode_modifier_delta(np_out: np.ndarray, effect: slai.EffectModifierDelta) -> None:
    np_out[_OFF_MODIFIER_KIND + _INDEX_MODIFIER_KIND[effect.kind]] = 1.0
    if effect.kind.is_buff:
        np_out[_OFF_MODIFIER_BUFF] = 1.0
    np_out[_OFF_STACKS] += effect.stacks


def _encode_modifier_multiply(np_out: np.ndarray, effect: slai.EffectModifierMultiply) -> None:
    np_out[_OFF_FACTOR] += effect.factor
    np_out[_OFF_MODIFIER_KIND + _INDEX_MODIFIER_KIND[effect.kind]] = 1.0
    if effect.kind.is_buff:
        np_out[_OFF_MODIFIER_BUFF] = 1.0


def _encode_monster_spawn(np_out: np.ndarray, effect: slai.EffectMonsterSpawn) -> None:
    np_out[_OFF_MONSTER_NAME + _VOCAB_MONSTER.get(effect.name, 0)] = 1.0


def _encode_reward_roll_cards(np_out: np.ndarray, effect: slai.EffectRewardRollCards) -> None:
    np_out[_OFF_COUNT] += effect.bundles
    np_out[_OFF_REWARD_ROLL_TRIGGER + _INDEX_REWARD_ROLL_TRIGGER[effect.trigger]] = 1.0


def _encode_monster_split(np_out: np.ndarray, effect: slai.EffectMonsterSplit) -> None:
    np_out[_OFF_MONSTER_NAME + _VOCAB_MONSTER.get(effect.name, 0)] = 1.0


def _encode_monster_summon(np_out: np.ndarray, effect: slai.EffectMonsterSummon) -> None:
    # Each name's share of the pool
    for name in effect.pool:
        np_out[_OFF_MONSTER_NAME + _VOCAB_MONSTER.get(name, 0)] += 1 / len(effect.pool)


def _encode_modifier_remove(np_out: np.ndarray, effect: slai.EffectModifierRemove) -> None:
    np_out[_OFF_MODIFIER_KIND + _INDEX_MODIFIER_KIND[effect.kind]] = 1.0
    np_out[_OFF_MODIFIER_BUFF] = float(effect.kind.is_buff)


def _encode_hexaghost_burn_increase(
    np_out: np.ndarray, effect: slai.EffectHexaghostBurnIncrease
) -> None:
    np_out[_OFF_COUNT] += effect.count


def _encode_neow_cards_roll(np_out: np.ndarray, effect: slai.EffectNeowCardsRoll) -> None:
    if effect.colorless:
        np_out[_OFF_COLORLESS] = 1.0
    if effect.rare_only:
        np_out[_OFF_RARE_ONLY] = 1.0


def _encode_potion_add_random(np_out: np.ndarray, effect: slai.EffectPotionAddRandom) -> None:
    if effect.limited:
        np_out[_OFF_LIMITED] = 1.0
    if effect.uniform:
        np_out[_OFF_UNIFORM] = 1.0


def _encode_relic_grant_pool(np_out: np.ndarray, effect: slai.EffectRelicGrantPool) -> None:
    for name in effect.pool:
        np_out[_OFF_RELIC_NAME + _VOCAB_RELIC.get(name, 0)] = 1.0
    np_out[_OFF_RELIC_POOL_SIZE] += len(effect.pool)


def _encode_relic_grant_random(np_out: np.ndarray, effect: slai.EffectRelicGrantRandom) -> None:
    np_out[_OFF_RELIC_EXCLUSION + _INDEX_RELIC_EXCLUSION[effect.exclusion]] = 1.0
    if effect.tier is not None:
        np_out[_OFF_RELIC_TIER + _INDEX_RELIC_TIER[effect.tier]] = 1.0


def _encode_relic_grant_specific(
    np_out: np.ndarray, effect: slai.EffectRelicGrantSpecific
) -> None:
    np_out[_OFF_RELIC_NAME + _VOCAB_RELIC.get(effect.name, 0)] = 1.0


def _encode_reward_roll_gold(np_out: np.ndarray, effect: slai.EffectRewardRollGold) -> None:
    _write_amount(np_out, effect.amount, _OFF_GOLD_ABSOLUTE, _OFF_GOLD_RELATIVE)


def _encode_reward_roll_potions(np_out: np.ndarray, effect: slai.EffectRewardRollPotions) -> None:
    np_out[_OFF_COUNT] += effect.count
    np_out[_OFF_REWARD_ROLL_TRIGGER + _INDEX_REWARD_ROLL_TRIGGER[effect.trigger]] = 1.0


def _encode_reward_roll_relic(np_out: np.ndarray, effect: slai.EffectRewardRollRelic) -> None:
    for name in effect.pool:
        np_out[_OFF_RELIC_NAME + _VOCAB_RELIC.get(name, 0)] = 1.0
    np_out[_OFF_RELIC_POOL_SIZE] += len(effect.pool)


def _encode_ritual_dagger_proc(np_out: np.ndarray, effect: slai.EffectRitualDaggerProc) -> None:
    np_out[_OFF_DAMAGE] += effect.bump


def _encode_scrap_ooze_reach(np_out: np.ndarray, effect: slai.EffectScrapOozeReach) -> None:
    if effect.advance_on_miss:
        np_out[_OFF_ADVANCE_ON_MISS] = 1.0
    np_out[_OFF_CHANCE] += effect.chance


def _encode_set_cost_override(np_out: np.ndarray, effect: slai.EffectSetCostOverride) -> None:
    np_out[_OFF_COST] += effect.amount
    if effect.only_reduce:
        np_out[_OFF_ONLY_REDUCE] = 1.0
    if effect.random:
        np_out[_OFF_RANDOM] = 1.0
    np_out[_OFF_COST_SCOPE + _INDEX_COST_SCOPE[effect.scope]] = 1.0


def _encode_skewer_proc(np_out: np.ndarray, effect: slai.EffectSkewerProc) -> None:
    np_out[_OFF_DAMAGE] += effect.damage


def _encode_sneaky_strike_proc(np_out: np.ndarray, effect: slai.EffectSneakyStrikeProc) -> None:
    np_out[_OFF_ENERGY] += effect.energy


def _encode_storm_of_steel_proc(np_out: np.ndarray, effect: slai.EffectStormOfSteelProc) -> None:
    if effect.upgraded:
        np_out[_OFF_UPGRADED] = 1.0


def _encode_strength_lose_temp(np_out: np.ndarray, effect: slai.EffectStrengthLoseTemp) -> None:
    np_out[_OFF_STACKS] += effect.stacks


_ENCODERS: dict[type, Callable[[np.ndarray, Any], None]] = {
    slai.EffectBlockGain: _encode_block_gain,
    slai.EffectBombArm: _encode_bomb_arm,
    slai.EffectCardAdd: _encode_card_add,
    slai.EffectCardAddRandom: _encode_card_add_random,
    slai.EffectCardDiscoverSelect: _encode_card_discover_select,
    slai.EffectCardDiscoverRoll: _encode_card_discover_roll,
    slai.EffectCardDraw: _encode_card_draw,
    slai.EffectCardDrawIfNoAttacks: _encode_card_draw_if_no_attacks,
    slai.EffectCardDrawUpTo: _encode_card_draw_up_to,
    slai.EffectCardDuplicate: _encode_card_duplicate,
    slai.EffectCardMove: _encode_card_move,
    slai.EffectCardSetupSelect: _encode_card_setup_select,
    slai.EffectCardTransform: _encode_card_transform,
    slai.EffectCombatEnd: _encode_combat_end,
    slai.EffectDamageFinisher: _encode_damage_finisher,
    slai.EffectDamageFlechettes: _encode_damage_flechettes,
    slai.EffectDamageMindBlast: _encode_damage_mind_blast,
    slai.EffectDamagePhysical: _encode_damage_physical,
    slai.EffectEnergyDelta: _encode_energy_delta,
    slai.EffectEscapePlanCheck: _encode_escape_plan_check,
    slai.EffectEventAdvanceState: _encode_event_advance_state,
    slai.EffectGamble: _encode_gamble,
    slai.EffectGlassKnifeDecay: _encode_glass_knife_decay,
    slai.EffectGoldDelta: _encode_gold_delta,
    slai.EffectHandOfGreedProc: _encode_hand_of_greed_proc,
    slai.EffectHealthDelta: _encode_health_delta,
    slai.EffectHealthSet: _encode_health_set,
    slai.EffectHeelHookProc: _encode_heel_hook_proc,
    slai.EffectJoustBet: _encode_joust_bet,
    slai.EffectMaxHealthDelta: _encode_max_health_delta,
    slai.EffectModifierDelta: _encode_modifier_delta,
    slai.EffectModifierMultiply: _encode_modifier_multiply,
    slai.EffectMonsterSpawn: _encode_monster_spawn,
    slai.EffectMonsterSplit: _encode_monster_split,
    slai.EffectMonsterSummon: _encode_monster_summon,
    slai.EffectRewardRollCards: _encode_reward_roll_cards,
    slai.EffectDamageDeal: _encode_damage_deal,
    slai.EffectModifierRemove: _encode_modifier_remove,
    slai.EffectHexaghostBurnIncrease: _encode_hexaghost_burn_increase,
    slai.EffectNeowCardsRoll: _encode_neow_cards_roll,
    slai.EffectPotionAddRandom: _encode_potion_add_random,
    slai.EffectRelicGrantPool: _encode_relic_grant_pool,
    slai.EffectRelicGrantRandom: _encode_relic_grant_random,
    slai.EffectRelicGrantSpecific: _encode_relic_grant_specific,
    slai.EffectRewardRollGold: _encode_reward_roll_gold,
    slai.EffectRewardRollPotions: _encode_reward_roll_potions,
    slai.EffectRewardRollRelic: _encode_reward_roll_relic,
    slai.EffectRitualDaggerProc: _encode_ritual_dagger_proc,
    slai.EffectScrapOozeReach: _encode_scrap_ooze_reach,
    slai.EffectSetCostOverride: _encode_set_cost_override,
    slai.EffectSkewerProc: _encode_skewer_proc,
    slai.EffectSneakyStrikeProc: _encode_sneaky_strike_proc,
    slai.EffectStormOfSteelProc: _encode_storm_of_steel_proc,
    slai.EffectStrengthLoseTemp: _encode_strength_lose_temp,
}


def _encode_effect_row_into(effect: Effect, np_out: np.ndarray) -> None:
    """Write one Effect's RAW (un-normalized) features into `np_out`.

    Parameters
    ----------
    effect : Effect
        The Effect to encode.
    np_out : np.ndarray
        (ENCODING_DIM_EFFECT,) float32. A zeroed row, written in place.
    """

    # Effect's kind
    np_out[_OFF_EFFECT_KIND + _INDEX_EFFECT[type(effect)]] = 1.0

    # Rows the engine never targets carry no `target` field at all
    target = getattr(effect, "target", None)
    if target is not None:
        np_out[_OFF_TARGETED] = 1.0
        np_out[_OFF_CANDIDATE_POOL + _INDEX_CAND_POOL[target.candidate_pool]] = 1.0
        np_out[_OFF_SELECTION_KIND + _INDEX_SELECTION_KIND[type(target.selection_kind)]] = 1.0
        np_out[_OFF_SELECT_COUNT] = float(getattr(target.selection_kind, "count", 1))

        # Multi-hot over the target's filters; an empty list keeps every candidate
        for candidate_filter in target.filters:
            np_out[_OFF_CANDIDATE_FILTER + _INDEX_CAND_FILTER[candidate_filter]] = 1.0

    # Effect Type encoder
    encode = _ENCODERS.get(type(effect))
    if encode is not None:
        encode(np_out, effect)

    # Buckets read the final raw floats; nothing may write those channels below
    for off_float, off_bucket, cap in _BUCKET_META:
        value = np_out[off_float]
        if value:
            np_out[off_bucket + int(math.sqrt(min(abs(value), cap)))] = 1.0

    # Buckets are magnitude-only; keep negative stacks (Malaise) linearly separable
    np_out[_OFF_STACKS_NEGATIVE] = float(np_out[_OFF_STACKS] < 0)


def make_effect_row(effect: Effect, effect_source: EffectSource) -> np.ndarray:
    """Make an Effect's row: its features, scaled, and its source's one-hot.

    Parameters
    ----------
    effect : Effect
        The Effect.
    effect_source : EffectSource
        What applies the Effect.

    Returns
    -------
    np.ndarray
        (ENCODING_DIM_EFFECT,) float32. The row.
    """
    np_row = np.zeros(ENCODING_DIM_EFFECT, dtype=np.float32)
    _encode_effect_row_into(effect, np_row)
    np_row[_OFF_EFFECT_SOURCE + effect_source] = 1.0
    _normalize_effect_row(np_row)

    return np_row


def _normalize_effect_row(np_effect_row: np.ndarray) -> None:
    """Scale the `_NORM_META` channels in place.

    Each is clipped to ±its cap, then divided by the cap or sqrt-compressed.

    Fancy indexing copies, so the transform runs on the gathered channels and the final
    assignment scatters them back.

    Parameters
    ----------
    np_effect_row : np.ndarray
        (ENCODING_DIM_EFFECT,) float32. A raw Effect row.
    """
    np_effect_gather = np_effect_row[_NP_IDX_SCALED]
    np.clip(np_effect_gather, -_NP_SCALED_CAP, _NP_SCALED_CAP, out=np_effect_gather)
    np_effect_gather[_NP_SCALED_LINEAR] /= _NP_SCALED_CAP[_NP_SCALED_LINEAR]
    np_value_sqrt = np_effect_gather[_NP_SCALED_SQRT]
    np_effect_gather[_NP_SCALED_SQRT] = np.sign(np_value_sqrt) * np.sqrt(
        np.abs(np_value_sqrt) / _NP_SCALED_CAP[_NP_SCALED_SQRT]
    )
    np_effect_row[_NP_IDX_SCALED] = np_effect_gather
