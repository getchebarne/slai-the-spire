"""Feature layout and static catalogs for effect rows."""

import math
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from enum import StrEnum
from enum import auto
from types import UnionType
from typing import Any, Protocol, SupportsInt, TypeVar, get_args

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


def amount_parts(amount: Amount) -> tuple[float, float, bool]:
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


E = TypeVar("E")
V = TypeVar("V")


class _MirrorEnum(SupportsInt, Protocol):
    @staticmethod
    def members() -> "list[Any]":
        ...


M = TypeVar("M", bound=_MirrorEnum)


def _field(cls: type[E], get: Callable[[E], V]) -> Iterator[V]:
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
                value_scalar = amount_parts(value)[0]
            else:
                value_scalar = float(value)

            max_ = max(max_, abs(value_scalar))

    return max_


def _vocabulary(*specs: Iterable[M | list[M]]) -> dict[M, int]:
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

    return {name: i + 1 for i, name in enumerate(sorted(names, key=int))}


def _index_enum(enum: type[M]) -> dict[M, int]:
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
    return {cls: i for i, cls in enumerate(get_args(union))}


INDEX_EFFECT = _index_union(Effect)
INDEX_CAND_POOL = _index_enum(CandidatePool)
INDEX_SELECTION_KIND = _index_union(SelectionKind)
INDEX_CAND_FILTER = _index_enum(CandidateFilter)
INDEX_MODIFIER_KIND = _index_enum(ModifierKind)
INDEX_REWARD_ROLL_TRIGGER = _index_enum(RewardRollTrigger)
INDEX_DELTA_SIGN = _index_enum(DeltaSign)
INDEX_CARD_PILE = _index_enum(CardPile)
INDEX_CARD_KIND = _index_enum(CardKind)
INDEX_CARD_COLOR = _index_enum(CardColor)
INDEX_CARD_RARITY = _index_enum(CardRarity)
INDEX_RELIC_TIER = _index_enum(RelicTier)
INDEX_RELIC_EXCLUSION = _index_enum(RelicExclusion)
INDEX_COST_SCOPE = _index_enum(CostScope)
INDEX_READ_AT = _index_enum(ReadAt)
INDEX_ROUNDING = _index_enum(Rounding)

VOCAB_CARD = _vocabulary(_field(slai.EffectCardAdd, lambda effect: effect.card_name))
VOCAB_RELIC = _vocabulary(
    _field(slai.EffectRelicGrantSpecific, lambda effect: effect.name),
    _field(slai.EffectRelicGrantPool, lambda effect: effect.pool),
    _field(slai.EffectRewardRollRelic, lambda effect: effect.pool),
)
VOCAB_MONSTER = _vocabulary(
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
_CAP_RELIC_POOL_SIZE = float(max(len(VOCAB_RELIC), 1))


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
OFF_EFFECT_KIND = _alloc("effect_kind", len(INDEX_EFFECT))

# Target
OFF_TARGETED = _alloc("targeted")
OFF_CANDIDATE_POOL = _alloc("candidate_pool", len(INDEX_CAND_POOL))
OFF_SELECTION_KIND = _alloc("selection_kind", len(INDEX_SELECTION_KIND))
OFF_SELECT_COUNT = _alloc("select_count")
OFF_CANDIDATE_FILTER = _alloc("candidate_filter", len(INDEX_CAND_FILTER))
OFF_EFFECT_SOURCE = _alloc(
    "effect_source", len(EffectSource)
)  # Written by encode_effect_row_into_w_cache, outside the row cache

# Categorical params, named for the engine type they one-hot
OFF_MODIFIER_KIND = _alloc("modifier_kind", len(INDEX_MODIFIER_KIND))
OFF_MODIFIER_BUFF = _alloc("modifier_buff")
OFF_DELTA_SIGN = _alloc("delta_sign", len(INDEX_DELTA_SIGN))
OFF_CARD_PILE = _alloc("card_pile", len(INDEX_CARD_PILE))
OFF_CARD_KIND = _alloc("card_kind", len(INDEX_CARD_KIND))
OFF_CARD_COLOR = _alloc("card_color", len(INDEX_CARD_COLOR))
OFF_CARD_RARITY = _alloc("card_rarity", len(INDEX_CARD_RARITY))
OFF_RELIC_TIER = _alloc("relic_tier", len(INDEX_RELIC_TIER))
OFF_RELIC_EXCLUSION = _alloc("relic_exclusion", len(INDEX_RELIC_EXCLUSION))
OFF_COST_SCOPE = _alloc("cost_scope", len(INDEX_COST_SCOPE))
OFF_COST_ZERO_PRESENT = _alloc("cost_zero_present")
OFF_REWARD_ROLL_TRIGGER = _alloc("reward_roll_trigger", len(INDEX_REWARD_ROLL_TRIGGER))
OFF_READ_AT = _alloc("read_at", len(INDEX_READ_AT))
OFF_ROUNDING = _alloc("rounding", len(INDEX_ROUNDING))

# Identity vocabularies, named for the engine name enum (index 0 = unseen)
OFF_CARD_NAME = _alloc("card_name", len(VOCAB_CARD) + 1)
OFF_RELIC_NAME = _alloc("relic_name", len(VOCAB_RELIC) + 1)
OFF_MONSTER_NAME = _alloc("monster_name", len(VOCAB_MONSTER) + 1)
OFF_RELIC_POOL_SIZE = _alloc("relic_pool_size")

# Magnitudes, one per physical unit
OFF_DAMAGE = _alloc("damage")
_OFF_DAMAGE_BUCKETS = _alloc("damage_buckets", _bucket_width(_CAP_DAMAGE))
OFF_INSTANCES = _alloc("instances")
OFF_BLOCK = _alloc("block")
OFF_HEALTH_ABSOLUTE = _alloc("health_absolute")
_OFF_HEALTH_BUCKETS = _alloc("health_buckets", _bucket_width(_CAP_HEALTH))
OFF_HEALTH_RELATIVE = _alloc("health_relative")
OFF_GOLD_ABSOLUTE = _alloc("gold_absolute")
_OFF_GOLD_BUCKETS = _alloc("gold_buckets", _bucket_width(_CAP_GOLD))
OFF_GOLD_RELATIVE = _alloc("gold_relative")
OFF_AMOUNT_IS_RANGE = _alloc("amount_is_range")
OFF_ENERGY = _alloc("energy")
OFF_STACKS = _alloc("stacks")
OFF_STACKS_NEGATIVE = _alloc("stacks_negative")
_OFF_STACKS_BUCKETS = _alloc("stacks_buckets", _bucket_width(_CAP_STACKS))
OFF_COUNT = _alloc("count")
OFF_DELTA = _alloc("delta")
OFF_CHANCE = _alloc("chance")
OFF_FACTOR = _alloc("factor")
OFF_COST = _alloc("cost")
OFF_TURNS = _alloc("turns")

# Boolean flags
# Boolean flags, named for the engine field
OFF_ADVANCE_ON_MISS = _alloc("advance_on_miss")
OFF_BLOCK_FROM_HIT = _alloc("block_from_hit")
OFF_BOTTOM = _alloc("bottom")
OFF_CHOOSE_DISCARDS = _alloc("choose_discards")
OFF_COLORLESS = _alloc("colorless")
OFF_ESCAPED_CHARACTER = _alloc("escaped_character")
OFF_FREE = _alloc("free")
OFF_IF_POISONED = _alloc("if_poisoned")
OFF_LIFESTEAL = _alloc("lifesteal")
OFF_LIMITED = _alloc("limited")
OFF_ON_OWNER = _alloc("on_owner")
OFF_ONLY_REDUCE = _alloc("only_reduce")
OFF_RANDOM = _alloc("random")
OFF_RARE_ONLY = _alloc("rare_only")
OFF_UNIFORM = _alloc("uniform")
OFF_UPGRADED = _alloc("upgraded")

# (float channel, bucket block, cap) — buckets derive from the final raw floats
BUCKET_META: tuple[tuple[int, int, float], ...] = (
    (OFF_DAMAGE, _OFF_DAMAGE_BUCKETS, _CAP_DAMAGE),
    (OFF_HEALTH_ABSOLUTE, _OFF_HEALTH_BUCKETS, _CAP_HEALTH),
    (OFF_GOLD_ABSOLUTE, _OFF_GOLD_BUCKETS, _CAP_GOLD),
    (OFF_STACKS, _OFF_STACKS_BUCKETS, _CAP_STACKS),
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
    (OFF_SELECT_COUNT, _CAP_SELECT_COUNT, _NormMode.LINEAR),
    (OFF_RELIC_POOL_SIZE, _CAP_RELIC_POOL_SIZE, _NormMode.LINEAR),
    (OFF_DAMAGE, _CAP_DAMAGE, _NormMode.SQRT),
    (OFF_INSTANCES, _CAP_INSTANCES, _NormMode.LINEAR),
    (OFF_BLOCK, _CAP_BLOCK, _NormMode.LINEAR),
    (OFF_HEALTH_ABSOLUTE, _CAP_HEALTH, _NormMode.SQRT),
    (OFF_GOLD_ABSOLUTE, _CAP_GOLD, _NormMode.SQRT),
    (OFF_ENERGY, _CAP_ENERGY, _NormMode.LINEAR),
    (OFF_STACKS, _CAP_STACKS, _NormMode.SQRT),
    (OFF_COUNT, _CAP_COUNT, _NormMode.LINEAR),
    (OFF_DELTA, _CAP_DELTA, _NormMode.LINEAR),
    (OFF_CHANCE, 100.0, _NormMode.LINEAR),
    (OFF_FACTOR, _CAP_FACTOR, _NormMode.LINEAR),
    (OFF_COST, _CAP_COST, _NormMode.LINEAR),
    (OFF_TURNS, _CAP_TURNS, _NormMode.LINEAR),
)

_SCALED_MASK = np.zeros(ENCODING_DIM_EFFECT, dtype=bool)
_CAPS = np.ones(ENCODING_DIM_EFFECT, dtype=np.float32)
_SQRT_MASK = np.zeros(ENCODING_DIM_EFFECT, dtype=bool)
for _off, _cap_value, _mode in _NORM_META:
    _SCALED_MASK[_off] = True
    _CAPS[_off] = _cap_value
    _SQRT_MASK[_off] = _mode == _NormMode.SQRT

# Only the _NORM_META channels need scaling at all
IDXS_SCALED = np.nonzero(_SCALED_MASK)[0]
SCALED_CAPS = _CAPS[IDXS_SCALED]
SCALED_SQRT = _SQRT_MASK[IDXS_SCALED]
SCALED_LINEAR = ~SCALED_SQRT
