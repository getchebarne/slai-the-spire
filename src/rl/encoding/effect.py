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
- **Normalization is vectorized and scoped** — see `normalize_effect_rows`.
"""

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
import torch
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
from slai import GameState
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
from src.rl.encoding.sparse import compile_sparse_rows
from src.rl.types import EffectSource
from src.rl.types import Jagged
from src.rl.types import X_SparseRows
from src.rl.types import make_channel_allocator


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
                value_scalar = _amount_parts(value)[0]
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
    return {member: int(member) for member in enum.members()}


def _index_union(union: UnionType) -> dict[type, int]:
    return {cls: i for i, cls in enumerate(get_args(union))}


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
class EffectChannel(StrEnum):
    # One-hot over the Effect union
    EFFECT_KIND = auto()

    # Target
    TARGETED = auto()
    CANDIDATE_POOL = auto()
    SELECTION_KIND = auto()
    SELECT_COUNT = auto()
    CANDIDATE_FILTER = auto()
    EFFECT_SOURCE = auto()  # Written by encode_effect_row_into_w_cache, outside the row cache

    # Categorical params, named for the engine type they one-hot
    MODIFIER_KIND = auto()
    MODIFIER_BUFF = auto()
    DELTA_SIGN = auto()
    CARD_PILE = auto()
    CARD_KIND = auto()
    CARD_COLOR = auto()
    CARD_RARITY = auto()
    RELIC_TIER = auto()
    RELIC_EXCLUSION = auto()
    COST_SCOPE = auto()
    COST_ZERO_PRESENT = auto()
    REWARD_ROLL_TRIGGER = auto()
    READ_AT = auto()
    ROUNDING = auto()

    # Identity vocabularies, named for the engine name enum
    CARD_NAME = auto()
    RELIC_NAME = auto()
    MONSTER_NAME = auto()
    RELIC_POOL_SIZE = auto()

    # Magnitudes, one per physical unit
    DAMAGE = auto()
    DAMAGE_BUCKETS = auto()
    INSTANCES = auto()
    BLOCK = auto()
    HEALTH_ABSOLUTE = auto()
    HEALTH_BUCKETS = auto()
    HEALTH_RELATIVE = auto()
    GOLD_ABSOLUTE = auto()
    GOLD_BUCKETS = auto()
    GOLD_RELATIVE = auto()
    AMOUNT_IS_RANGE = auto()
    ENERGY = auto()
    STACKS = auto()
    STACKS_NEGATIVE = auto()
    STACKS_BUCKETS = auto()
    COUNT = auto()
    DELTA = auto()
    CHANCE = auto()
    FACTOR = auto()
    COST = auto()
    TURNS = auto()

    # Boolean flags, named for the engine field
    ADVANCE_ON_MISS = auto()
    BLOCK_FROM_HIT = auto()
    BOTTOM = auto()
    CHOOSE_DISCARDS = auto()
    COLORLESS = auto()
    ESCAPED_CHARACTER = auto()
    FREE = auto()
    IF_POISONED = auto()
    LIFESTEAL = auto()
    LIMITED = auto()
    ON_OWNER = auto()
    ONLY_REDUCE = auto()
    RANDOM = auto()
    RARE_ONLY = auto()
    UNIFORM = auto()
    UPGRADED = auto()


def _bucket_width(cap: float) -> int:
    return int(math.sqrt(cap)) + 1


CHANNEL_LAYOUT_EFFECT, _alloc = make_channel_allocator()


# Effect's kind
OFF_EFFECT_KIND = _alloc(EffectChannel.EFFECT_KIND, len(_INDEX_EFFECT))

# Target
OFF_TARGETED = _alloc(EffectChannel.TARGETED)
OFF_CANDIDATE_POOL = _alloc(EffectChannel.CANDIDATE_POOL, len(_INDEX_CAND_POOL))
OFF_SELECTION_KIND = _alloc(EffectChannel.SELECTION_KIND, len(_INDEX_SELECTION_KIND))
OFF_SELECT_COUNT = _alloc(EffectChannel.SELECT_COUNT)
OFF_CANDIDATE_FILTER = _alloc(EffectChannel.CANDIDATE_FILTER, len(_INDEX_CAND_FILTER))
OFF_EFFECT_SOURCE = _alloc(EffectChannel.EFFECT_SOURCE, len(EffectSource))

# Categorical params
OFF_MODIFIER_KIND = _alloc(EffectChannel.MODIFIER_KIND, len(_INDEX_MODIFIER_KIND))
OFF_MODIFIER_BUFF = _alloc(EffectChannel.MODIFIER_BUFF)
OFF_DELTA_SIGN = _alloc(EffectChannel.DELTA_SIGN, len(_INDEX_DELTA_SIGN))
OFF_CARD_PILE = _alloc(EffectChannel.CARD_PILE, len(_INDEX_CARD_PILE))
OFF_CARD_KIND = _alloc(EffectChannel.CARD_KIND, len(_INDEX_CARD_KIND))
OFF_CARD_COLOR = _alloc(EffectChannel.CARD_COLOR, len(_INDEX_CARD_COLOR))
OFF_CARD_RARITY = _alloc(EffectChannel.CARD_RARITY, len(_INDEX_CARD_RARITY))
OFF_RELIC_TIER = _alloc(EffectChannel.RELIC_TIER, len(_INDEX_RELIC_TIER))
OFF_RELIC_EXCLUSION = _alloc(EffectChannel.RELIC_EXCLUSION, len(_INDEX_RELIC_EXCLUSION))
OFF_COST_SCOPE = _alloc(EffectChannel.COST_SCOPE, len(_INDEX_COST_SCOPE))
OFF_COST_ZERO_PRESENT = _alloc(EffectChannel.COST_ZERO_PRESENT)
OFF_REWARD_ROLL_TRIGGER = _alloc(
    EffectChannel.REWARD_ROLL_TRIGGER, len(_INDEX_REWARD_ROLL_TRIGGER)
)
OFF_READ_AT = _alloc(EffectChannel.READ_AT, len(_INDEX_READ_AT))
OFF_ROUNDING = _alloc(EffectChannel.ROUNDING, len(_INDEX_ROUNDING))

# Identity vocabularies (index 0 = unseen)
OFF_CARD_NAME = _alloc(EffectChannel.CARD_NAME, len(_VOCAB_CARD) + 1)
OFF_RELIC_NAME = _alloc(EffectChannel.RELIC_NAME, len(_VOCAB_RELIC) + 1)
OFF_MONSTER_NAME = _alloc(EffectChannel.MONSTER_NAME, len(_VOCAB_MONSTER) + 1)
OFF_RELIC_POOL_SIZE = _alloc(EffectChannel.RELIC_POOL_SIZE)

# Magnitudes
OFF_DAMAGE = _alloc(EffectChannel.DAMAGE)
OFF_DAMAGE_BUCKETS = _alloc(EffectChannel.DAMAGE_BUCKETS, _bucket_width(_CAP_DAMAGE))
OFF_INSTANCES = _alloc(EffectChannel.INSTANCES)
OFF_BLOCK = _alloc(EffectChannel.BLOCK)
OFF_HEALTH_ABSOLUTE = _alloc(EffectChannel.HEALTH_ABSOLUTE)
OFF_HEALTH_BUCKETS = _alloc(EffectChannel.HEALTH_BUCKETS, _bucket_width(_CAP_HEALTH))
OFF_HEALTH_RELATIVE = _alloc(EffectChannel.HEALTH_RELATIVE)
OFF_GOLD_ABSOLUTE = _alloc(EffectChannel.GOLD_ABSOLUTE)
OFF_GOLD_BUCKETS = _alloc(EffectChannel.GOLD_BUCKETS, _bucket_width(_CAP_GOLD))
OFF_GOLD_RELATIVE = _alloc(EffectChannel.GOLD_RELATIVE)
OFF_AMOUNT_IS_RANGE = _alloc(EffectChannel.AMOUNT_IS_RANGE)
OFF_ENERGY = _alloc(EffectChannel.ENERGY)
OFF_STACKS = _alloc(EffectChannel.STACKS)
OFF_STACKS_NEGATIVE = _alloc(EffectChannel.STACKS_NEGATIVE)
OFF_STACKS_BUCKETS = _alloc(EffectChannel.STACKS_BUCKETS, _bucket_width(_CAP_STACKS))
OFF_COUNT = _alloc(EffectChannel.COUNT)
OFF_DELTA = _alloc(EffectChannel.DELTA)
OFF_CHANCE = _alloc(EffectChannel.CHANCE)
OFF_FACTOR = _alloc(EffectChannel.FACTOR)
OFF_COST = _alloc(EffectChannel.COST)
OFF_TURNS = _alloc(EffectChannel.TURNS)

# Boolean flags
OFF_ADVANCE_ON_MISS = _alloc(EffectChannel.ADVANCE_ON_MISS)
OFF_BLOCK_FROM_HIT = _alloc(EffectChannel.BLOCK_FROM_HIT)
OFF_BOTTOM = _alloc(EffectChannel.BOTTOM)
OFF_CHOOSE_DISCARDS = _alloc(EffectChannel.CHOOSE_DISCARDS)
OFF_COLORLESS = _alloc(EffectChannel.COLORLESS)
OFF_ESCAPED_CHARACTER = _alloc(EffectChannel.ESCAPED_CHARACTER)
OFF_FREE = _alloc(EffectChannel.FREE)
OFF_IF_POISONED = _alloc(EffectChannel.IF_POISONED)
OFF_LIFESTEAL = _alloc(EffectChannel.LIFESTEAL)
OFF_LIMITED = _alloc(EffectChannel.LIMITED)
OFF_ON_OWNER = _alloc(EffectChannel.ON_OWNER)
OFF_ONLY_REDUCE = _alloc(EffectChannel.ONLY_REDUCE)
OFF_RANDOM = _alloc(EffectChannel.RANDOM)
OFF_RARE_ONLY = _alloc(EffectChannel.RARE_ONLY)
OFF_UNIFORM = _alloc(EffectChannel.UNIFORM)
OFF_UPGRADED = _alloc(EffectChannel.UPGRADED)

# (float channel, bucket block, cap) — buckets derive from the final raw floats
_BUCKET_META: tuple[tuple[int, int, float], ...] = (
    (OFF_DAMAGE, OFF_DAMAGE_BUCKETS, _CAP_DAMAGE),
    (OFF_HEALTH_ABSOLUTE, OFF_HEALTH_BUCKETS, _CAP_HEALTH),
    (OFF_GOLD_ABSOLUTE, OFF_GOLD_BUCKETS, _CAP_GOLD),
    (OFF_STACKS, OFF_STACKS_BUCKETS, _CAP_STACKS),
)

# Total
ENCODING_DIM_EFFECT = sum(w for _, w in CHANNEL_LAYOUT_EFFECT)

# The channels that set several of a row's columns
_MULTI_HOT_CHANNELS = (
    EffectChannel.CANDIDATE_FILTER,
    EffectChannel.RELIC_NAME,
    EffectChannel.MONSTER_NAME,
)

# Most nonzero columns an Effect row holds: one per one-hot or scalar channel, and every column
# of the multi-hot ones
ENCODING_DIM_SPARSE_EFFECT = sum(
    width if channel in _MULTI_HOT_CHANNELS else 1 for channel, width in CHANNEL_LAYOUT_EFFECT
)


# Per-effect-kind encoders
def _write_amount(out: np.ndarray, amount: Amount, off_abs: int, off_rel: int) -> None:
    """Write an Amount union into absolute/relative channels, plus the shared range flag.

    Parameters
    ----------
    out : np.ndarray
        (ENCODING_DIM_EFFECT,) float32. The Effect's raw row, updated in place.
    amount : Amount
        The amount; a relative one also writes its ReadAt and Rounding one-hots.
    off_abs : int
        The absolute channel's column.
    off_rel : int
        The relative channel's column.
    """
    absolute, relative, is_range = _amount_parts(amount)
    out[off_abs] += absolute
    out[off_rel] += relative
    if is_range:
        out[OFF_AMOUNT_IS_RANGE] = 1.0

    # A fraction also says which value it reads and how it rounds
    if isinstance(amount, AmountRelative):
        out[OFF_READ_AT + _INDEX_READ_AT[amount.read_at]] = 1.0
        out[OFF_ROUNDING + _INDEX_ROUNDING[amount.rounding]] = 1.0


def _cost_zero(out: np.ndarray, value: CostScope | None) -> None:
    """Write an Optional[CostScope]: a presence flag plus the scope one-hot.

    Parameters
    ----------
    out : np.ndarray
        (ENCODING_DIM_EFFECT,) float32. The Effect's raw row, updated in place.
    value : CostScope | None
        The Effect's `cost_zero` scope; None writes nothing.
    """
    if value is not None:
        out[OFF_COST_ZERO_PRESENT] = 1.0
        out[OFF_COST_SCOPE + _INDEX_COST_SCOPE[value]] = 1.0


def _encode_block_gain(out: np.ndarray, effect: slai.EffectBlockGain) -> None:
    out[OFF_BLOCK] += effect.amount


def _encode_bomb_arm(out: np.ndarray, effect: slai.EffectBombArm) -> None:
    out[OFF_DAMAGE] += effect.damage
    out[OFF_TURNS] += effect.turns


def _encode_card_add(out: np.ndarray, effect: slai.EffectCardAdd) -> None:
    out[OFF_CARD_NAME + _VOCAB_CARD.get(effect.card_name, 0)] = 1.0
    out[OFF_COUNT] += effect.count
    out[OFF_CARD_PILE + _INDEX_CARD_PILE[effect.card_pile]] = 1.0
    if effect.upgraded:
        out[OFF_UPGRADED] = 1.0


def _encode_card_add_random(out: np.ndarray, effect: slai.EffectCardAddRandom) -> None:
    out[OFF_CARD_COLOR + _INDEX_CARD_COLOR[effect.color]] = 1.0
    _cost_zero(out, effect.cost_zero)
    out[OFF_COUNT] += effect.count
    if effect.kind is not None:
        out[OFF_CARD_KIND + _INDEX_CARD_KIND[effect.kind]] = 1.0
    out[OFF_CARD_PILE + _INDEX_CARD_PILE[effect.card_pile]] = 1.0
    if effect.rarity is not None:
        out[OFF_CARD_RARITY + _INDEX_CARD_RARITY[effect.rarity]] = 1.0
    if effect.upgraded:
        out[OFF_UPGRADED] = 1.0


def _encode_card_discover_select(out: np.ndarray, effect: slai.EffectCardDiscoverSelect) -> None:
    out[OFF_COUNT] += effect.copies
    _cost_zero(out, effect.cost_zero)
    out[OFF_CARD_PILE + _INDEX_CARD_PILE[effect.card_pile]] = 1.0


def _encode_card_discover_roll(out: np.ndarray, effect: slai.EffectCardDiscoverRoll) -> None:
    out[OFF_CARD_COLOR + _INDEX_CARD_COLOR[effect.color]] = 1.0
    out[OFF_COUNT] += effect.count
    if effect.kind is not None:
        out[OFF_CARD_KIND + _INDEX_CARD_KIND[effect.kind]] = 1.0


def _encode_card_draw(out: np.ndarray, effect: slai.EffectCardDraw) -> None:
    out[OFF_COUNT] += effect.count


def _encode_card_draw_if_no_attacks(
    out: np.ndarray, effect: slai.EffectCardDrawIfNoAttacks
) -> None:
    out[OFF_COUNT] += effect.count


def _encode_card_draw_up_to(out: np.ndarray, effect: slai.EffectCardDrawUpTo) -> None:
    out[OFF_COUNT] += effect.amount


def _encode_card_duplicate(out: np.ndarray, effect: slai.EffectCardDuplicate) -> None:
    out[OFF_CARD_PILE + _INDEX_CARD_PILE[effect.card_pile]] = 1.0


def _encode_card_move(out: np.ndarray, effect: slai.EffectCardMove) -> None:
    _cost_zero(out, effect.cost_zero)
    out[OFF_CARD_PILE + _INDEX_CARD_PILE[effect.card_pile]] = 1.0


def _encode_card_setup_select(out: np.ndarray, effect: slai.EffectCardSetupSelect) -> None:
    if effect.bottom:
        out[OFF_BOTTOM] = 1.0
    if effect.free:
        out[OFF_FREE] = 1.0


def _encode_card_transform(out: np.ndarray, effect: slai.EffectCardTransform) -> None:
    if effect.upgraded:
        out[OFF_UPGRADED] = 1.0


def _encode_combat_end(out: np.ndarray, effect: slai.EffectCombatEnd) -> None:
    if effect.escaped_character:
        out[OFF_ESCAPED_CHARACTER] = 1.0


def _encode_damage_finisher(out: np.ndarray, effect: slai.EffectDamageFinisher) -> None:
    out[OFF_DAMAGE] += effect.damage


def _encode_damage_flechettes(out: np.ndarray, effect: slai.EffectDamageFlechettes) -> None:
    out[OFF_DAMAGE] += effect.damage


def _encode_damage_mind_blast(out: np.ndarray, effect: slai.EffectDamageMindBlast) -> None:
    out[OFF_DAMAGE] += effect.damage


def _encode_damage_physical(out: np.ndarray, effect: slai.EffectDamagePhysical) -> None:
    out[OFF_DAMAGE] += effect.amount
    if effect.block_from_hit:
        out[OFF_BLOCK_FROM_HIT] = 1.0
    if effect.if_poisoned:
        out[OFF_IF_POISONED] = 1.0
    out[OFF_INSTANCES] += effect.instances
    if effect.lifesteal:
        out[OFF_LIFESTEAL] = 1.0


def _encode_damage_deal(out: np.ndarray, effect: slai.EffectDamageDeal) -> None:
    out[OFF_DAMAGE] += effect.amount
    if effect.lifesteal:
        out[OFF_LIFESTEAL] = 1.0


def _encode_energy_delta(out: np.ndarray, effect: slai.EffectEnergyDelta) -> None:
    out[OFF_ENERGY] += effect.amount
    out[OFF_DELTA_SIGN + _INDEX_DELTA_SIGN[effect.sign]] = 1.0


def _encode_escape_plan_check(out: np.ndarray, effect: slai.EffectEscapePlanCheck) -> None:
    out[OFF_BLOCK] += effect.block


def _encode_event_advance_state(out: np.ndarray, effect: slai.EffectEventAdvanceState) -> None:
    out[OFF_DELTA] += effect.delta


def _encode_gamble(out: np.ndarray, effect: slai.EffectGamble) -> None:
    if effect.choose_discards:
        out[OFF_CHOOSE_DISCARDS] = 1.0


def _encode_glass_knife_decay(out: np.ndarray, effect: slai.EffectGlassKnifeDecay) -> None:
    out[OFF_DAMAGE] += effect.delta


def _encode_gold_delta(out: np.ndarray, effect: slai.EffectGoldDelta) -> None:
    _write_amount(out, effect.amount, OFF_GOLD_ABSOLUTE, OFF_GOLD_RELATIVE)
    out[OFF_DELTA_SIGN + _INDEX_DELTA_SIGN[effect.sign]] = 1.0


def _encode_hand_of_greed_proc(out: np.ndarray, effect: slai.EffectHandOfGreedProc) -> None:
    out[OFF_GOLD_ABSOLUTE] += effect.gold


def _encode_health_delta(out: np.ndarray, effect: slai.EffectHealthDelta) -> None:
    _write_amount(out, effect.amount, OFF_HEALTH_ABSOLUTE, OFF_HEALTH_RELATIVE)
    out[OFF_DELTA_SIGN + _INDEX_DELTA_SIGN[effect.sign]] = 1.0


def _encode_health_set(out: np.ndarray, effect: slai.EffectHealthSet) -> None:
    _write_amount(out, effect.amount, OFF_HEALTH_ABSOLUTE, OFF_HEALTH_RELATIVE)


def _encode_heel_hook_proc(out: np.ndarray, effect: slai.EffectHeelHookProc) -> None:
    out[OFF_DAMAGE] += effect.damage


def _encode_joust_bet(out: np.ndarray, effect: slai.EffectJoustBet) -> None:
    if effect.on_owner:
        out[OFF_ON_OWNER] = 1.0


def _encode_max_health_delta(out: np.ndarray, effect: slai.EffectMaxHealthDelta) -> None:
    _write_amount(out, effect.amount, OFF_HEALTH_ABSOLUTE, OFF_HEALTH_RELATIVE)
    out[OFF_DELTA_SIGN + _INDEX_DELTA_SIGN[effect.sign]] = 1.0


def _encode_modifier_delta(out: np.ndarray, effect: slai.EffectModifierDelta) -> None:
    out[OFF_MODIFIER_KIND + _INDEX_MODIFIER_KIND[effect.kind]] = 1.0
    if effect.kind.is_buff:
        out[OFF_MODIFIER_BUFF] = 1.0
    out[OFF_STACKS] += effect.stacks


def _encode_modifier_multiply(out: np.ndarray, effect: slai.EffectModifierMultiply) -> None:
    out[OFF_FACTOR] += effect.factor
    out[OFF_MODIFIER_KIND + _INDEX_MODIFIER_KIND[effect.kind]] = 1.0
    if effect.kind.is_buff:
        out[OFF_MODIFIER_BUFF] = 1.0


def _encode_monster_spawn(out: np.ndarray, effect: slai.EffectMonsterSpawn) -> None:
    out[OFF_MONSTER_NAME + _VOCAB_MONSTER.get(effect.name, 0)] = 1.0


def _encode_reward_roll_cards(out: np.ndarray, effect: slai.EffectRewardRollCards) -> None:
    out[OFF_COUNT] += effect.bundles
    out[OFF_REWARD_ROLL_TRIGGER + _INDEX_REWARD_ROLL_TRIGGER[effect.trigger]] = 1.0


def _encode_monster_split(out: np.ndarray, effect: slai.EffectMonsterSplit) -> None:
    out[OFF_MONSTER_NAME + _VOCAB_MONSTER.get(effect.name, 0)] = 1.0


def _encode_monster_summon(out: np.ndarray, effect: slai.EffectMonsterSummon) -> None:
    # Each name's share of the pool
    for name in effect.pool:
        out[OFF_MONSTER_NAME + _VOCAB_MONSTER.get(name, 0)] += 1 / len(effect.pool)


def _encode_modifier_remove(out: np.ndarray, effect: slai.EffectModifierRemove) -> None:
    out[OFF_MODIFIER_KIND + _INDEX_MODIFIER_KIND[effect.kind]] = 1.0
    out[OFF_MODIFIER_BUFF] = float(effect.kind.is_buff)


def _encode_hexaghost_burn_increase(
    out: np.ndarray, effect: slai.EffectHexaghostBurnIncrease
) -> None:
    out[OFF_COUNT] += effect.count


def _encode_neow_cards_roll(out: np.ndarray, effect: slai.EffectNeowCardsRoll) -> None:
    if effect.colorless:
        out[OFF_COLORLESS] = 1.0
    if effect.rare_only:
        out[OFF_RARE_ONLY] = 1.0


def _encode_potion_add_random(out: np.ndarray, effect: slai.EffectPotionAddRandom) -> None:
    if effect.limited:
        out[OFF_LIMITED] = 1.0
    if effect.uniform:
        out[OFF_UNIFORM] = 1.0


def _encode_relic_grant_pool(out: np.ndarray, effect: slai.EffectRelicGrantPool) -> None:
    for name in effect.pool:
        out[OFF_RELIC_NAME + _VOCAB_RELIC.get(name, 0)] = 1.0
    out[OFF_RELIC_POOL_SIZE] += len(effect.pool)


def _encode_relic_grant_random(out: np.ndarray, effect: slai.EffectRelicGrantRandom) -> None:
    out[OFF_RELIC_EXCLUSION + _INDEX_RELIC_EXCLUSION[effect.exclusion]] = 1.0
    if effect.tier is not None:
        out[OFF_RELIC_TIER + _INDEX_RELIC_TIER[effect.tier]] = 1.0


def _encode_relic_grant_specific(out: np.ndarray, effect: slai.EffectRelicGrantSpecific) -> None:
    out[OFF_RELIC_NAME + _VOCAB_RELIC.get(effect.name, 0)] = 1.0


def _encode_reward_roll_gold(out: np.ndarray, effect: slai.EffectRewardRollGold) -> None:
    _write_amount(out, effect.amount, OFF_GOLD_ABSOLUTE, OFF_GOLD_RELATIVE)


def _encode_reward_roll_potions(out: np.ndarray, effect: slai.EffectRewardRollPotions) -> None:
    out[OFF_COUNT] += effect.count
    out[OFF_REWARD_ROLL_TRIGGER + _INDEX_REWARD_ROLL_TRIGGER[effect.trigger]] = 1.0


def _encode_reward_roll_relic(out: np.ndarray, effect: slai.EffectRewardRollRelic) -> None:
    for name in effect.pool:
        out[OFF_RELIC_NAME + _VOCAB_RELIC.get(name, 0)] = 1.0
    out[OFF_RELIC_POOL_SIZE] += len(effect.pool)


def _encode_ritual_dagger_proc(out: np.ndarray, effect: slai.EffectRitualDaggerProc) -> None:
    out[OFF_DAMAGE] += effect.bump


def _encode_scrap_ooze_reach(out: np.ndarray, effect: slai.EffectScrapOozeReach) -> None:
    if effect.advance_on_miss:
        out[OFF_ADVANCE_ON_MISS] = 1.0
    out[OFF_CHANCE] += effect.chance


def _encode_set_cost_override(out: np.ndarray, effect: slai.EffectSetCostOverride) -> None:
    out[OFF_COST] += effect.amount
    if effect.only_reduce:
        out[OFF_ONLY_REDUCE] = 1.0
    if effect.random:
        out[OFF_RANDOM] = 1.0
    out[OFF_COST_SCOPE + _INDEX_COST_SCOPE[effect.scope]] = 1.0


def _encode_skewer_proc(out: np.ndarray, effect: slai.EffectSkewerProc) -> None:
    out[OFF_DAMAGE] += effect.damage


def _encode_sneaky_strike_proc(out: np.ndarray, effect: slai.EffectSneakyStrikeProc) -> None:
    out[OFF_ENERGY] += effect.energy


def _encode_storm_of_steel_proc(out: np.ndarray, effect: slai.EffectStormOfSteelProc) -> None:
    if effect.upgraded:
        out[OFF_UPGRADED] = 1.0


def _encode_strength_lose_temp(out: np.ndarray, effect: slai.EffectStrengthLoseTemp) -> None:
    out[OFF_STACKS] += effect.stacks


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


def encode_effect_row_into(effect: Effect, out: np.ndarray) -> None:
    """Write one Effect's RAW (un-normalized) features into `out`.

    Callers that write many rows should normalize the whole block once with
    `normalize_effect_rows`.

    Parameters
    ----------
    effect : Effect
        The Effect to encode.
    out : np.ndarray
        (ENCODING_DIM_EFFECT,) float32. A zeroed row, written in place.
    """

    # Effect's kind
    out[OFF_EFFECT_KIND + _INDEX_EFFECT[type(effect)]] = 1.0

    # Rows the engine never targets carry no `target` field at all
    target = getattr(effect, "target", None)
    if target is not None:
        out[OFF_TARGETED] = 1.0
        out[OFF_CANDIDATE_POOL + _INDEX_CAND_POOL[target.candidate_pool]] = 1.0
        out[OFF_SELECTION_KIND + _INDEX_SELECTION_KIND[type(target.selection_kind)]] = 1.0
        out[OFF_SELECT_COUNT] = float(getattr(target.selection_kind, "count", 1))

        # Multi-hot over the target's filters; an empty list keeps every candidate
        for candidate_filter in target.filters:
            out[OFF_CANDIDATE_FILTER + _INDEX_CAND_FILTER[candidate_filter]] = 1.0

    # Effect Type encoder
    encode = _ENCODERS.get(type(effect))
    if encode is not None:
        encode(out, effect)

    # Buckets read the final raw floats; nothing may write those channels below
    for off_float, off_bucket, cap in _BUCKET_META:
        value = out[off_float]
        if value:
            out[off_bucket + int(math.sqrt(min(abs(value), cap)))] = 1.0

    # Buckets are magnitude-only; keep negative stacks (Malaise) linearly separable
    out[OFF_STACKS_NEGATIVE] = float(out[OFF_STACKS] < 0)


# Per-effect RAW row cache. Effect pyclasses are frozen and hash by field
# value, so an Effect object is its own value-key
_EFFECT_ROW_CACHE: dict[Effect, np.ndarray] = {}
_EFFECT_ROW_CACHE_MAX = 100_000


def effect_key(effects: Iterable[Effect]) -> tuple[Effect, ...]:
    """Return a value-key over an Effect list (frozen pyclasses hash by field).

    Parameters
    ----------
    effects : Iterable[Effect]
        The Effects to key.

    Returns
    -------
    tuple[Effect, ...]
        The Effects, in order.
    """
    return tuple(effects)


def encode_effect_row_into_w_cache(effect: Effect, source: EffectSource, out: np.ndarray) -> None:
    row = _EFFECT_ROW_CACHE.get(effect)
    if row is None:
        row = np.zeros(ENCODING_DIM_EFFECT, dtype=np.float32)
        encode_effect_row_into(effect, row)
        row.flags.writeable = False  # Guard the cached master copy
        if len(_EFFECT_ROW_CACHE) >= _EFFECT_ROW_CACHE_MAX:
            _EFFECT_ROW_CACHE.clear()
        _EFFECT_ROW_CACHE[effect] = row
    out[:] = row
    out[OFF_EFFECT_SOURCE + source] = 1.0


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
_IDXS_SCALED = np.nonzero(_SCALED_MASK)[0]
_SCALED_CAPS = _CAPS[_IDXS_SCALED]
_SCALED_SQRT = _SQRT_MASK[_IDXS_SCALED]
_SCALED_LINEAR = ~_SCALED_SQRT


def normalize_effect_rows(rows: np.ndarray) -> None:
    """Scale the `_NORM_META` channels in place.

    Each is clipped to ±its cap, then divided by the cap or sqrt-compressed.

    Run it once per block, over all its rows: per row, it costs far more. Fancy indexing copies,
    so the transform runs on a gathered block and the final assignment scatters it back; `...`
    keeps it rank-agnostic.

    Parameters
    ----------
    rows : np.ndarray
        (..., ENCODING_DIM_EFFECT) float32. Raw Effect rows: any array whose LAST axis is the
        channel axis.
    """
    if not rows.size:
        return

    rows_gather = rows[..., _IDXS_SCALED]
    np.clip(rows_gather, -_SCALED_CAPS, _SCALED_CAPS, out=rows_gather)
    rows_gather[..., _SCALED_LINEAR] /= _SCALED_CAPS[_SCALED_LINEAR]
    sqrt_vals = rows_gather[..., _SCALED_SQRT]
    rows_gather[..., _SCALED_SQRT] = np.sign(sqrt_vals) * np.sqrt(
        np.abs(sqrt_vals) / _SCALED_CAPS[_SCALED_SQRT]
    )
    rows[..., _IDXS_SCALED] = rows_gather


def encode_batch_effect_pending(
    batch_game_state: list[GameState], device: torch.device
) -> Jagged[X_SparseRows]:
    """Encode the pending Effect rows as one (P, D) block, held sparse.

    One row per state with a halt pending, every state's after the previous state's.

    Parameters
    ----------
    batch_game_state : list[GameState]
        The states to encode, B of them.
    device : torch.device
        The device of the returned tensors.

    Returns
    -------
    Jagged[X_SparseRows]
        The pending Effect rows, held sparse, and each state's (B,) row count, 1 or 0.
    """
    batch_effect_pending = [game_state.effect_pending for game_state in batch_game_state]
    np_count = np.array(
        [effect_pending is not None for effect_pending in batch_effect_pending], dtype=np.int64
    )
    np_effect_rows = np.zeros((np_count.sum(), ENCODING_DIM_EFFECT), dtype=np.float32)
    row = 0
    for effect_pending in batch_effect_pending:
        if effect_pending is not None:
            encode_effect_row_into_w_cache(
                effect_pending, EffectSource.PENDING, np_effect_rows[row]
            )
            row += 1

    normalize_effect_rows(np_effect_rows)
    return compile_sparse_rows(
        np_effect_rows, np_count, ENCODING_DIM_SPARSE_EFFECT, device
    )
