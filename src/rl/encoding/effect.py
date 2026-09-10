"""Per-effect feature rows.

Each `Effect` becomes one factored row — kind OHE + target features + typed
param channels + magnitudes on shared physical scales — embedded model-side by
one shared MLP and sum-pooled into its owning entity's token. That replaces the
old per-entity histogram, whose disjoint bins re-learned the same semantics per
family (modifier side, CardAdd pile, selection kind) and buried magnitudes in
entity-summed cells.

Two rules the layout depends on:

- **Referents contribute identity, never effects.** `CardAdd{Bite}` writes a
  name-vocabulary column, not Bite's own encoding. That makes the row depth-0 by
  construction, which matters because the engine's effect-reference graph
  contains a self-loop (Endless Agony adds itself).
- **Normalization is vectorized and scoped** — see `normalize_effect_rows`.
"""

import math
import warnings
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Sequence
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
from src.rl.constants import MAX_EFFECTS_STATE
from src.rl.constants import MAX_SIZE_CARD_PILE_DECK
from src.rl.constants import MAX_SIZE_CARD_PILE_HAND
from src.rl.constants import MULTI_STAB_INSTANCES_MAX
from src.rl.types import OWNER_PENDING
from src.rl.types import EffectSource
from src.rl.types import EntityKind
from src.rl.types import X_Effects
from src.rl.types import make_channel_allocator


# Live card/relic amounts arrive modifier-scaled, so a catalog maximum is a
# floor rather than a ceiling; past this the channel clamps
_LIVE_SCALE_HEADROOM = 1.5
_ASCENSIONS = range(ASCENSION_LEVEL + 1)


def catalog_rows() -> Iterator[tuple[str, Effect]]:
    """(label, effect) for every effect reachable from a static definition, at
    every ascension. Labelled because verify_caps_drift reports which definition
    set each channel's peak."""
    # Cards
    for template in get_card_templates():
        label = f"card {template.name}{'+' if template.upgraded else ''}"
        for effect in (*template.effects_play, *template.effects_discard, *template.effects_draw):
            yield label, effect

    # Potions
    for potion in get_potion_templates():
        for effect in potion.effects:
            yield f"potion {potion.name}", effect

    # Relics, one sweep per trigger field
    for relic in get_relic_templates():
        for trigger, effects in (
            ("combat_start", relic.effects_combat_start),
            ("turn_start", relic.effects_turn_start),
            ("turn_end", relic.effects_turn_end),
            ("combat_end", relic.effects_combat_end),
            ("pickup", relic.effects_pickup),
            ("rest", relic.effects_rest),
            ("counter", relic.effects_counter),
        ):
            for effect in effects:
                yield f"relic {relic.name} {trigger}", effect

    # Event Options
    for ascension in _ASCENSIONS:
        for option in get_event_option_templates(ascension):
            for effect in option.effects:
                yield f"event {option.event} a{ascension}", effect

    # Monster moves
    for ascension in _ASCENSIONS:
        for monster in get_monster_templates(ascension):
            for move in monster.moves:
                for effect in move.effects:
                    yield f"monster {monster.name} {move.name} a{ascension}", effect


# Bucketed by exact type once
_BY_TYPE: dict = {}
for _, _effect in catalog_rows():
    _BY_TYPE.setdefault(type(_effect), []).append(_effect)
del _effect


def _amount_parts(amount: Amount) -> tuple[float, float, bool]:
    """Amount union -> (absolute, relative_fraction, is_range). A range folds to
    its midpoint in the absolute channel."""
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
    """One field across every catalog instance of a kind. The typed accessor is
    the point: mypy checks each `lambda e: e.x` against the kind's stub."""
    effects: list[E] = _BY_TYPE.get(cls, [])
    for effect in effects:
        yield get(effect)


def _cap(*specs: Iterable[Amount | float], floor: float = 1.0) -> float:
    """Max |value| over catalog fields sharing a physical scale."""
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
    """Catalog-derived identity vocabulary; index 0 is reserved as unseen."""
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

_VOCAB_CARD = _vocabulary(_field(slai.EffectCardAdd, lambda e: e.card_name))
_VOCAB_RELIC = _vocabulary(
    _field(slai.EffectRelicGrantSpecific, lambda e: e.name),
    _field(slai.EffectRelicGrantPool, lambda e: e.pool),
    _field(slai.EffectRewardRollRelic, lambda e: e.pool),
)
_VOCAB_MONSTER = _vocabulary(
    _field(slai.EffectMonsterSpawn, lambda e: e.name),
    _field(slai.EffectMonsterSplit, lambda e: e.name),
)


# Caps
_CAP_DAMAGE = _LIVE_SCALE_HEADROOM * _cap(
    _field(slai.EffectDamagePhysical, lambda e: e.amount),
    _field(slai.EffectDamagePhysicalIfPoisoned, lambda e: e.amount),
    _field(slai.EffectDamageFinisher, lambda e: e.damage),
    _field(slai.EffectDamageFlechettes, lambda e: e.damage),
    _field(slai.EffectDamageMindBlast, lambda e: e.damage),
    _field(slai.EffectBombArm, lambda e: e.damage),
    _field(slai.EffectRitualDaggerProc, lambda e: e.bump),
    _field(slai.EffectGlassKnifeDecay, lambda e: e.delta),
)
# The catalog's Multi-Stab is a 0-hit placeholder, so its live hit count is bridged
_CAP_INSTANCES = max(
    _cap(_field(slai.EffectDamagePhysical, lambda e: e.instances)),
    float(MULTI_STAB_INSTANCES_MAX),
)
_CAP_BLOCK = _LIVE_SCALE_HEADROOM * _cap(
    _field(slai.EffectBlockGain, lambda e: e.amount),
    _field(slai.EffectEscapePlanCheck, lambda e: e.block),
)
_CAP_HEALTH = _cap(
    _field(slai.EffectHealthDelta, lambda e: e.amount),
    _field(slai.EffectMaxHealthDelta, lambda e: e.amount),
)
_CAP_GOLD = _cap(
    _field(slai.EffectGoldDelta, lambda e: e.amount),
    _field(slai.EffectHandOfGreedProc, lambda e: e.gold),
    _field(slai.EffectRewardRollGold, lambda e: e.amount),
)
_CAP_ENERGY = _cap(
    _field(slai.EffectEnergyDelta, lambda e: e.amount),
    _field(slai.EffectSneakyStrikeProc, lambda e: e.energy),
)
_CAP_STACKS = _cap(
    _field(slai.EffectModifierDelta, lambda e: e.stacks),
    _field(slai.EffectStrengthLoseTemp, lambda e: e.stacks),
)
_CAP_COUNT = max(
    _cap(
        _field(slai.EffectCardAdd, lambda e: e.count),
        _field(slai.EffectCardAddRandom, lambda e: e.count),
        _field(slai.EffectCardDiscoverPick, lambda e: e.copies),
        _field(slai.EffectCardDraw, lambda e: e.count),
        _field(slai.EffectCardDrawIfNoAttacks, lambda e: e.count),
        _field(slai.EffectCardDrawUpTo, lambda e: e.amount),
        _field(slai.EffectCardDiscoverRoll, lambda e: e.count),
        _field(slai.EffectRewardRollPotions, lambda e: e.count),
    ),
    float(MAX_SIZE_CARD_PILE_HAND),
)
_CAP_DELTA = _cap(_field(slai.EffectEventAdvanceState, lambda e: e.delta))
_CAP_FACTOR = _cap(_field(slai.EffectModifierMultiply, lambda e: e.factor))
_CAP_COST = _cap(_field(slai.EffectSetCostOverride, lambda e: e.amount))
_CAP_TURNS = float(slai.GameEnv.BOMB_FUSE_TURNS)
_CAP_SELECT_COUNT = float(MAX_SIZE_CARD_PILE_DECK)
_CAP_RELIC_POOL_SIZE = float(max(len(_VOCAB_RELIC), 1))


# Contiguous groups; every column carries a cap and a mode
class Channel(StrEnum):
    # One-hot over the Effect union
    EFFECT_KIND = auto()

    # Target
    TARGETED = auto()
    CANDIDATE_POOL = auto()
    SELECTION_KIND = auto()
    SELECT_COUNT = auto()
    CANDIDATE_FILTER = auto()
    EFFECT_SOURCE = auto()  # written by state.py, not the row encoders

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
    BOTTOM = auto()
    CHOOSE_DISCARDS = auto()
    ESCAPED_CHARACTER = auto()
    FREE = auto()
    LIFESTEAL = auto()
    LIMITED = auto()
    ON_OWNER = auto()
    ONLY_REDUCE = auto()
    RANDOM = auto()
    UNIFORM = auto()
    UPGRADED = auto()


def _bucket_width(cap: float) -> int:
    return int(math.sqrt(cap)) + 1


CHANNEL_LAYOUT, _alloc = make_channel_allocator()


# Effect's kind
OFF_EFFECT_KIND = _alloc(Channel.EFFECT_KIND, len(_INDEX_EFFECT))

# Target
OFF_TARGETED = _alloc(Channel.TARGETED)
OFF_CANDIDATE_POOL = _alloc(Channel.CANDIDATE_POOL, len(_INDEX_CAND_POOL))
OFF_SELECTION_KIND = _alloc(Channel.SELECTION_KIND, len(_INDEX_SELECTION_KIND))
OFF_SELECT_COUNT = _alloc(Channel.SELECT_COUNT)
OFF_CANDIDATE_FILTER = _alloc(Channel.CANDIDATE_FILTER, len(_INDEX_CAND_FILTER))
OFF_EFFECT_SOURCE = _alloc(Channel.EFFECT_SOURCE, len(EffectSource))

# Categorical params
OFF_MODIFIER_KIND = _alloc(Channel.MODIFIER_KIND, len(_INDEX_MODIFIER_KIND))
OFF_MODIFIER_BUFF = _alloc(Channel.MODIFIER_BUFF)
OFF_DELTA_SIGN = _alloc(Channel.DELTA_SIGN, len(_INDEX_DELTA_SIGN))
OFF_CARD_PILE = _alloc(Channel.CARD_PILE, len(_INDEX_CARD_PILE))
OFF_CARD_KIND = _alloc(Channel.CARD_KIND, len(_INDEX_CARD_KIND))
OFF_CARD_COLOR = _alloc(Channel.CARD_COLOR, len(_INDEX_CARD_COLOR))
OFF_CARD_RARITY = _alloc(Channel.CARD_RARITY, len(_INDEX_CARD_RARITY))
OFF_RELIC_TIER = _alloc(Channel.RELIC_TIER, len(_INDEX_RELIC_TIER))
OFF_RELIC_EXCLUSION = _alloc(Channel.RELIC_EXCLUSION, len(_INDEX_RELIC_EXCLUSION))
OFF_COST_SCOPE = _alloc(Channel.COST_SCOPE, len(_INDEX_COST_SCOPE))
OFF_COST_ZERO_PRESENT = _alloc(Channel.COST_ZERO_PRESENT)
OFF_REWARD_ROLL_TRIGGER = _alloc(Channel.REWARD_ROLL_TRIGGER, len(_INDEX_REWARD_ROLL_TRIGGER))
OFF_READ_AT = _alloc(Channel.READ_AT, len(_INDEX_READ_AT))
OFF_ROUNDING = _alloc(Channel.ROUNDING, len(_INDEX_ROUNDING))

# Identity vocabularies (index 0 = unseen)
OFF_CARD_NAME = _alloc(Channel.CARD_NAME, len(_VOCAB_CARD) + 1)
OFF_RELIC_NAME = _alloc(Channel.RELIC_NAME, len(_VOCAB_RELIC) + 1)
OFF_MONSTER_NAME = _alloc(Channel.MONSTER_NAME, len(_VOCAB_MONSTER) + 1)
OFF_RELIC_POOL_SIZE = _alloc(Channel.RELIC_POOL_SIZE)

# Magnitudes
OFF_DAMAGE = _alloc(Channel.DAMAGE)
OFF_DAMAGE_BUCKETS = _alloc(Channel.DAMAGE_BUCKETS, _bucket_width(_CAP_DAMAGE))
OFF_INSTANCES = _alloc(Channel.INSTANCES)
OFF_BLOCK = _alloc(Channel.BLOCK)
OFF_HEALTH_ABSOLUTE = _alloc(Channel.HEALTH_ABSOLUTE)
OFF_HEALTH_BUCKETS = _alloc(Channel.HEALTH_BUCKETS, _bucket_width(_CAP_HEALTH))
OFF_HEALTH_RELATIVE = _alloc(Channel.HEALTH_RELATIVE)
OFF_GOLD_ABSOLUTE = _alloc(Channel.GOLD_ABSOLUTE)
OFF_GOLD_BUCKETS = _alloc(Channel.GOLD_BUCKETS, _bucket_width(_CAP_GOLD))
OFF_GOLD_RELATIVE = _alloc(Channel.GOLD_RELATIVE)
OFF_AMOUNT_IS_RANGE = _alloc(Channel.AMOUNT_IS_RANGE)
OFF_ENERGY = _alloc(Channel.ENERGY)
OFF_STACKS = _alloc(Channel.STACKS)
OFF_STACKS_NEGATIVE = _alloc(Channel.STACKS_NEGATIVE)
OFF_STACKS_BUCKETS = _alloc(Channel.STACKS_BUCKETS, _bucket_width(_CAP_STACKS))
OFF_COUNT = _alloc(Channel.COUNT)
OFF_DELTA = _alloc(Channel.DELTA)
OFF_CHANCE = _alloc(Channel.CHANCE)
OFF_FACTOR = _alloc(Channel.FACTOR)
OFF_COST = _alloc(Channel.COST)
OFF_TURNS = _alloc(Channel.TURNS)

# Boolean flags
OFF_ADVANCE_ON_MISS = _alloc(Channel.ADVANCE_ON_MISS)
OFF_BOTTOM = _alloc(Channel.BOTTOM)
OFF_CHOOSE_DISCARDS = _alloc(Channel.CHOOSE_DISCARDS)
OFF_ESCAPED_CHARACTER = _alloc(Channel.ESCAPED_CHARACTER)
OFF_FREE = _alloc(Channel.FREE)
OFF_LIFESTEAL = _alloc(Channel.LIFESTEAL)
OFF_LIMITED = _alloc(Channel.LIMITED)
OFF_ON_OWNER = _alloc(Channel.ON_OWNER)
OFF_ONLY_REDUCE = _alloc(Channel.ONLY_REDUCE)
OFF_RANDOM = _alloc(Channel.RANDOM)
OFF_UNIFORM = _alloc(Channel.UNIFORM)
OFF_UPGRADED = _alloc(Channel.UPGRADED)

# (float channel, bucket block, cap) — buckets derive from the final raw floats
_BUCKET_META: tuple[tuple[int, int, float], ...] = (
    (OFF_DAMAGE, OFF_DAMAGE_BUCKETS, _CAP_DAMAGE),
    (OFF_HEALTH_ABSOLUTE, OFF_HEALTH_BUCKETS, _CAP_HEALTH),
    (OFF_GOLD_ABSOLUTE, OFF_GOLD_BUCKETS, _CAP_GOLD),
    (OFF_STACKS, OFF_STACKS_BUCKETS, _CAP_STACKS),
)

# Total
ENCODING_DIM_EFFECT = sum(w for _, w in CHANNEL_LAYOUT)


# Per-kind encoders
def _write_amount(row: np.ndarray, value: Amount, off_abs: int, off_rel: int) -> None:
    """Amount union -> absolute/relative channels, plus the shared range flag."""
    absolute, relative, is_range = _amount_parts(value)
    row[off_abs] += absolute
    row[off_rel] += relative
    if is_range:
        row[OFF_AMOUNT_IS_RANGE] = 1.0
    # A fraction also says which value it reads and how it rounds
    if isinstance(value, AmountRelative):
        row[OFF_READ_AT + _INDEX_READ_AT[value.read_at]] = 1.0
        row[OFF_ROUNDING + _INDEX_ROUNDING[value.rounding]] = 1.0


def _cost_zero(row: np.ndarray, value: CostScope | None) -> None:
    """Optional[CostScope]: a presence flag plus the scope one-hot."""
    if value is not None:
        row[OFF_COST_ZERO_PRESENT] = 1.0
        row[OFF_COST_SCOPE + _INDEX_COST_SCOPE[value]] = 1.0


def _encode_block_gain(row: np.ndarray, e: slai.EffectBlockGain) -> None:
    row[OFF_BLOCK] += e.amount


def _encode_bomb_arm(row: np.ndarray, e: slai.EffectBombArm) -> None:
    row[OFF_DAMAGE] += e.damage
    row[OFF_TURNS] += e.turns


def _encode_card_add(row: np.ndarray, e: slai.EffectCardAdd) -> None:
    row[OFF_CARD_NAME + _VOCAB_CARD.get(e.card_name, 0)] = 1.0
    row[OFF_COUNT] += e.count
    row[OFF_CARD_PILE + _INDEX_CARD_PILE[e.card_pile]] = 1.0
    if e.upgraded:
        row[OFF_UPGRADED] = 1.0


def _encode_card_add_random(row: np.ndarray, e: slai.EffectCardAddRandom) -> None:
    row[OFF_CARD_COLOR + _INDEX_CARD_COLOR[e.color]] = 1.0
    _cost_zero(row, e.cost_zero)
    row[OFF_COUNT] += e.count
    if e.kind is not None:
        row[OFF_CARD_KIND + _INDEX_CARD_KIND[e.kind]] = 1.0
    row[OFF_CARD_PILE + _INDEX_CARD_PILE[e.card_pile]] = 1.0
    if e.rarity is not None:
        row[OFF_CARD_RARITY + _INDEX_CARD_RARITY[e.rarity]] = 1.0
    if e.upgraded:
        row[OFF_UPGRADED] = 1.0


def _encode_card_discover_pick(row: np.ndarray, e: slai.EffectCardDiscoverPick) -> None:
    row[OFF_COUNT] += e.copies
    _cost_zero(row, e.cost_zero)
    row[OFF_CARD_PILE + _INDEX_CARD_PILE[e.card_pile]] = 1.0


def _encode_card_discover_roll(row: np.ndarray, e: slai.EffectCardDiscoverRoll) -> None:
    row[OFF_CARD_COLOR + _INDEX_CARD_COLOR[e.color]] = 1.0
    row[OFF_COUNT] += e.count
    if e.kind is not None:
        row[OFF_CARD_KIND + _INDEX_CARD_KIND[e.kind]] = 1.0


def _encode_card_draw(row: np.ndarray, e: slai.EffectCardDraw) -> None:
    row[OFF_COUNT] += e.count


def _encode_card_draw_if_no_attacks(row: np.ndarray, e: slai.EffectCardDrawIfNoAttacks) -> None:
    row[OFF_COUNT] += e.count


def _encode_card_draw_up_to(row: np.ndarray, e: slai.EffectCardDrawUpTo) -> None:
    row[OFF_COUNT] += e.amount


def _encode_card_duplicate(row: np.ndarray, e: slai.EffectCardDuplicate) -> None:
    row[OFF_CARD_PILE + _INDEX_CARD_PILE[e.card_pile]] = 1.0


def _encode_card_move(row: np.ndarray, e: slai.EffectCardMove) -> None:
    _cost_zero(row, e.cost_zero)
    row[OFF_CARD_PILE + _INDEX_CARD_PILE[e.card_pile]] = 1.0


def _encode_card_setup_pick(row: np.ndarray, e: slai.EffectCardSetupPick) -> None:
    if e.bottom:
        row[OFF_BOTTOM] = 1.0
    if e.free:
        row[OFF_FREE] = 1.0


def _encode_card_transform(row: np.ndarray, e: slai.EffectCardTransform) -> None:
    if e.upgraded:
        row[OFF_UPGRADED] = 1.0


def _encode_combat_end(row: np.ndarray, e: slai.EffectCombatEnd) -> None:
    if e.escaped_character:
        row[OFF_ESCAPED_CHARACTER] = 1.0


def _encode_damage_finisher(row: np.ndarray, e: slai.EffectDamageFinisher) -> None:
    row[OFF_DAMAGE] += e.damage


def _encode_damage_flechettes(row: np.ndarray, e: slai.EffectDamageFlechettes) -> None:
    row[OFF_DAMAGE] += e.damage


def _encode_damage_mind_blast(row: np.ndarray, e: slai.EffectDamageMindBlast) -> None:
    row[OFF_DAMAGE] += e.damage


def _encode_damage_physical(row: np.ndarray, e: slai.EffectDamagePhysical) -> None:
    row[OFF_DAMAGE] += e.amount
    row[OFF_INSTANCES] += e.instances
    if e.lifesteal:
        row[OFF_LIFESTEAL] = 1.0


def _encode_damage_deal(row: np.ndarray, e: slai.EffectDamageDeal) -> None:
    row[OFF_DAMAGE] += e.amount
    if e.lifesteal:
        row[OFF_LIFESTEAL] = 1.0


def _encode_damage_physical_if_poisoned(
    row: np.ndarray, e: slai.EffectDamagePhysicalIfPoisoned
) -> None:
    row[OFF_DAMAGE] += e.amount


def _encode_energy_delta(row: np.ndarray, e: slai.EffectEnergyDelta) -> None:
    row[OFF_ENERGY] += e.amount
    row[OFF_DELTA_SIGN + _INDEX_DELTA_SIGN[e.sign]] = 1.0


def _encode_escape_plan_check(row: np.ndarray, e: slai.EffectEscapePlanCheck) -> None:
    row[OFF_BLOCK] += e.block


def _encode_event_advance_state(row: np.ndarray, e: slai.EffectEventAdvanceState) -> None:
    row[OFF_DELTA] += e.delta


def _encode_gamble(row: np.ndarray, e: slai.EffectGamble) -> None:
    if e.choose_discards:
        row[OFF_CHOOSE_DISCARDS] = 1.0


def _encode_glass_knife_decay(row: np.ndarray, e: slai.EffectGlassKnifeDecay) -> None:
    row[OFF_DAMAGE] += e.delta


def _encode_gold_delta(row: np.ndarray, e: slai.EffectGoldDelta) -> None:
    _write_amount(row, e.amount, OFF_GOLD_ABSOLUTE, OFF_GOLD_RELATIVE)
    row[OFF_DELTA_SIGN + _INDEX_DELTA_SIGN[e.sign]] = 1.0


def _encode_hand_of_greed_proc(row: np.ndarray, e: slai.EffectHandOfGreedProc) -> None:
    row[OFF_GOLD_ABSOLUTE] += e.gold


def _encode_health_delta(row: np.ndarray, e: slai.EffectHealthDelta) -> None:
    _write_amount(row, e.amount, OFF_HEALTH_ABSOLUTE, OFF_HEALTH_RELATIVE)
    row[OFF_DELTA_SIGN + _INDEX_DELTA_SIGN[e.sign]] = 1.0


def _encode_joust_bet(row: np.ndarray, e: slai.EffectJoustBet) -> None:
    if e.on_owner:
        row[OFF_ON_OWNER] = 1.0


def _encode_max_health_delta(row: np.ndarray, e: slai.EffectMaxHealthDelta) -> None:
    _write_amount(row, e.amount, OFF_HEALTH_ABSOLUTE, OFF_HEALTH_RELATIVE)
    row[OFF_DELTA_SIGN + _INDEX_DELTA_SIGN[e.sign]] = 1.0


def _encode_modifier_delta(row: np.ndarray, e: slai.EffectModifierDelta) -> None:
    row[OFF_MODIFIER_KIND + _INDEX_MODIFIER_KIND[e.kind]] = 1.0
    if e.kind.is_buff:
        row[OFF_MODIFIER_BUFF] = 1.0
    row[OFF_STACKS] += e.stacks


def _encode_modifier_multiply(row: np.ndarray, e: slai.EffectModifierMultiply) -> None:
    row[OFF_FACTOR] += e.factor
    row[OFF_MODIFIER_KIND + _INDEX_MODIFIER_KIND[e.kind]] = 1.0
    if e.kind.is_buff:
        row[OFF_MODIFIER_BUFF] = 1.0


def _encode_monster_spawn(row: np.ndarray, e: slai.EffectMonsterSpawn) -> None:
    row[OFF_MONSTER_NAME + _VOCAB_MONSTER.get(e.name, 0)] = 1.0


def _encode_reward_roll_cards(row: np.ndarray, e: slai.EffectRewardRollCards) -> None:
    row[OFF_COUNT] += e.bundles
    row[OFF_REWARD_ROLL_TRIGGER + _INDEX_REWARD_ROLL_TRIGGER[e.trigger]] = 1.0


def _encode_monster_split(row: np.ndarray, e: slai.EffectMonsterSplit) -> None:
    row[OFF_MONSTER_NAME + _VOCAB_MONSTER.get(e.name, 0)] = 1.0


def _encode_modifier_remove(row: np.ndarray, e: slai.EffectModifierRemove) -> None:
    row[OFF_MODIFIER_KIND + _INDEX_MODIFIER_KIND[e.kind]] = 1.0
    row[OFF_MODIFIER_BUFF] = float(e.kind.is_buff)


def _encode_hexaghost_burn_increase(row: np.ndarray, e: slai.EffectHexaghostBurnIncrease) -> None:
    row[OFF_COUNT] += e.count


def _encode_potion_add_random(row: np.ndarray, e: slai.EffectPotionAddRandom) -> None:
    if e.limited:
        row[OFF_LIMITED] = 1.0
    if e.uniform:
        row[OFF_UNIFORM] = 1.0


def _encode_relic_grant_pool(row: np.ndarray, e: slai.EffectRelicGrantPool) -> None:
    for name in e.pool:
        row[OFF_RELIC_NAME + _VOCAB_RELIC.get(name, 0)] = 1.0
    row[OFF_RELIC_POOL_SIZE] += len(e.pool)


def _encode_relic_grant_random(row: np.ndarray, e: slai.EffectRelicGrantRandom) -> None:
    row[OFF_RELIC_EXCLUSION + _INDEX_RELIC_EXCLUSION[e.exclusion]] = 1.0
    if e.tier is not None:
        row[OFF_RELIC_TIER + _INDEX_RELIC_TIER[e.tier]] = 1.0


def _encode_relic_grant_specific(row: np.ndarray, e: slai.EffectRelicGrantSpecific) -> None:
    row[OFF_RELIC_NAME + _VOCAB_RELIC.get(e.name, 0)] = 1.0


def _encode_reward_roll_gold(row: np.ndarray, e: slai.EffectRewardRollGold) -> None:
    _write_amount(row, e.amount, OFF_GOLD_ABSOLUTE, OFF_GOLD_RELATIVE)


def _encode_reward_roll_potions(row: np.ndarray, e: slai.EffectRewardRollPotions) -> None:
    row[OFF_COUNT] += e.count
    row[OFF_REWARD_ROLL_TRIGGER + _INDEX_REWARD_ROLL_TRIGGER[e.trigger]] = 1.0


def _encode_reward_roll_relic(row: np.ndarray, e: slai.EffectRewardRollRelic) -> None:
    for name in e.pool:
        row[OFF_RELIC_NAME + _VOCAB_RELIC.get(name, 0)] = 1.0
    row[OFF_RELIC_POOL_SIZE] += len(e.pool)


def _encode_ritual_dagger_proc(row: np.ndarray, e: slai.EffectRitualDaggerProc) -> None:
    row[OFF_DAMAGE] += e.bump


def _encode_scrap_ooze_reach(row: np.ndarray, e: slai.EffectScrapOozeReach) -> None:
    if e.advance_on_miss:
        row[OFF_ADVANCE_ON_MISS] = 1.0
    row[OFF_CHANCE] += e.chance


def _encode_set_cost_override(row: np.ndarray, e: slai.EffectSetCostOverride) -> None:
    row[OFF_COST] += e.amount
    if e.only_reduce:
        row[OFF_ONLY_REDUCE] = 1.0
    if e.random:
        row[OFF_RANDOM] = 1.0
    row[OFF_COST_SCOPE + _INDEX_COST_SCOPE[e.scope]] = 1.0


def _encode_sneaky_strike_proc(row: np.ndarray, e: slai.EffectSneakyStrikeProc) -> None:
    row[OFF_ENERGY] += e.energy


def _encode_storm_of_steel_proc(row: np.ndarray, e: slai.EffectStormOfSteelProc) -> None:
    if e.upgraded:
        row[OFF_UPGRADED] = 1.0


def _encode_strength_lose_temp(row: np.ndarray, e: slai.EffectStrengthLoseTemp) -> None:
    row[OFF_STACKS] += e.stacks


_EffectWriter = Callable[[np.ndarray, Any], None]
_ENCODERS: dict[type, _EffectWriter] = {
    slai.EffectBlockGain: _encode_block_gain,
    slai.EffectBombArm: _encode_bomb_arm,
    slai.EffectCardAdd: _encode_card_add,
    slai.EffectCardAddRandom: _encode_card_add_random,
    slai.EffectCardDiscoverPick: _encode_card_discover_pick,
    slai.EffectCardDiscoverRoll: _encode_card_discover_roll,
    slai.EffectCardDraw: _encode_card_draw,
    slai.EffectCardDrawIfNoAttacks: _encode_card_draw_if_no_attacks,
    slai.EffectCardDrawUpTo: _encode_card_draw_up_to,
    slai.EffectCardDuplicate: _encode_card_duplicate,
    slai.EffectCardMove: _encode_card_move,
    slai.EffectCardSetupPick: _encode_card_setup_pick,
    slai.EffectCardTransform: _encode_card_transform,
    slai.EffectCombatEnd: _encode_combat_end,
    slai.EffectDamageFinisher: _encode_damage_finisher,
    slai.EffectDamageFlechettes: _encode_damage_flechettes,
    slai.EffectDamageMindBlast: _encode_damage_mind_blast,
    slai.EffectDamagePhysical: _encode_damage_physical,
    slai.EffectDamagePhysicalIfPoisoned: _encode_damage_physical_if_poisoned,
    slai.EffectEnergyDelta: _encode_energy_delta,
    slai.EffectEscapePlanCheck: _encode_escape_plan_check,
    slai.EffectEventAdvanceState: _encode_event_advance_state,
    slai.EffectGamble: _encode_gamble,
    slai.EffectGlassKnifeDecay: _encode_glass_knife_decay,
    slai.EffectGoldDelta: _encode_gold_delta,
    slai.EffectHandOfGreedProc: _encode_hand_of_greed_proc,
    slai.EffectHealthDelta: _encode_health_delta,
    slai.EffectJoustBet: _encode_joust_bet,
    slai.EffectMaxHealthDelta: _encode_max_health_delta,
    slai.EffectModifierDelta: _encode_modifier_delta,
    slai.EffectModifierMultiply: _encode_modifier_multiply,
    slai.EffectMonsterSpawn: _encode_monster_spawn,
    slai.EffectMonsterSplit: _encode_monster_split,
    slai.EffectRewardRollCards: _encode_reward_roll_cards,
    slai.EffectDamageDeal: _encode_damage_deal,
    slai.EffectModifierRemove: _encode_modifier_remove,
    slai.EffectHexaghostBurnIncrease: _encode_hexaghost_burn_increase,
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
    slai.EffectSneakyStrikeProc: _encode_sneaky_strike_proc,
    slai.EffectStormOfSteelProc: _encode_storm_of_steel_proc,
    slai.EffectStrengthLoseTemp: _encode_strength_lose_temp,
}


def encode_effect_row_into(effect: Effect, row: np.ndarray) -> None:
    """Write one effect's RAW (un-normalized) features into `row`.

    Callers that write many rows should normalize the whole block once with
    `normalize_effect_rows`."""
    # Effect's kind
    row[OFF_EFFECT_KIND + _INDEX_EFFECT[type(effect)]] = 1.0

    # Rows the engine never targets carry no `target` field at all
    target = getattr(effect, "target", None)
    if target is not None:
        row[OFF_TARGETED] = 1.0
        row[OFF_CANDIDATE_POOL + _INDEX_CAND_POOL[target.candidate_pool]] = 1.0
        row[OFF_SELECTION_KIND + _INDEX_SELECTION_KIND[type(target.selection_kind)]] = 1.0
        row[OFF_SELECT_COUNT] = float(getattr(target.selection_kind, "count", 1))
        # Multi-hot over the target's filters; an empty list keeps every candidate
        for candidate_filter in target.filters:
            row[OFF_CANDIDATE_FILTER + _INDEX_CAND_FILTER[candidate_filter]] = 1.0

    encode = _ENCODERS.get(type(effect))
    if encode is not None:
        encode(row, effect)

    # Buckets read the final raw floats; nothing may write those channels below
    for off_float, off_bucket, cap in _BUCKET_META:
        value = row[off_float]
        if value:
            row[off_bucket + int(math.sqrt(min(abs(value), cap)))] = 1.0

    # Buckets are magnitude-only; keep negative stacks (Malaise) linearly separable
    row[OFF_STACKS_NEGATIVE] = float(row[OFF_STACKS] < 0)


# Per-effect RAW row cache. Effect pyclasses are frozen and hash by field
# value, so an Effect object is its own value-key
_EFFECT_ROW_CACHE: dict[Effect, np.ndarray] = {}
_EFFECT_ROW_CACHE_MAX = 100_000


def effect_key(effects: Iterable[Effect]) -> tuple[Effect, ...]:
    """Value-key over an effect list (frozen pyclasses hash by field)."""
    return tuple(effects)


def encode_effect_row_into_w_cache(effect: Effect, out: np.ndarray) -> None:
    row = _EFFECT_ROW_CACHE.get(effect)
    if row is None:
        row = np.zeros(ENCODING_DIM_EFFECT, dtype=np.float32)
        encode_effect_row_into(effect, row)
        row.flags.writeable = False  # guard the cached master copy
        if len(_EFFECT_ROW_CACHE) >= _EFFECT_ROW_CACHE_MAX:
            _EFFECT_ROW_CACHE.clear()
        _EFFECT_ROW_CACHE[effect] = row
    out[:] = row


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

# Only the 13 magnitude channels need scaling at all
_IDXS_SCALED = np.nonzero(_SCALED_MASK)[0]
_SCALED_CAPS = _CAPS[_IDXS_SCALED]
_SCALED_SQRT = _SQRT_MASK[_IDXS_SCALED]
_SCALED_LINEAR = ~_SCALED_SQRT


def normalize_effect_rows(rows: np.ndarray) -> None:
    """Scale every channel in place. `rows` is any array whose LAST axis is the
    channel axis — (n, D) or a (B, n, D) slice of the packed block.

    Two costs this shape avoids, both measured: running per row costs 34 numpy
    slice calls (~78 us) whatever the payload, and normalizing the full padded
    block instead of the caller's used prefix is 11x the work (132 vs 15
    us/state). Fancy indexing copies, so the transform runs on a gathered block
    and the final assignment scatters it back; `...` keeps it rank-agnostic."""
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


def encode_effect_rows(effects: list) -> np.ndarray:
    """Fold an effect list into an (n, ENCODING_DIM_EFFECT) float32 block."""
    rows = np.zeros((len(effects), ENCODING_DIM_EFFECT), dtype=np.float32)
    for i, effect in enumerate(effects):
        encode_effect_row_into_w_cache(effect, rows[i])

    normalize_effect_rows(rows)
    return rows


# Row families that have already warned about overflow — truncation is never silent
_WARNED_OVERFLOW: set[str] = set()


def encode_batch_effects(
    game_states: list[GameState],
    sources: Sequence[tuple[EntityKind, list[list[tuple[int, EffectSource, Effect]]]]],
    device: torch.device,
) -> X_Effects:
    """Every state's effect rows as one (B, E, D) block with their owner keys.

    `sources` is (owner kind, per-state (offset, source, effect) rows) in fill
    priority: the pending halt goes first, then the sources in order, so a state
    past MAX_EFFECTS_STATE drops its last sources' rows."""
    batch_size = len(game_states)
    np_effect = np.zeros((batch_size, MAX_EFFECTS_STATE, ENCODING_DIM_EFFECT), dtype=np.float32)
    np_entity_kind = np.full((batch_size, MAX_EFFECTS_STATE), -1, dtype=np.int64)
    np_offset = np.zeros((batch_size, MAX_EFFECTS_STATE), dtype=np.int64)

    idx_used = 0
    for b, game_state in enumerate(game_states):
        idx = 0

        # Pending Effect: owns no token
        pending = game_state.effect_pending
        if pending is not None:
            encode_effect_row_into_w_cache(pending, np_effect[b, idx])
            np_effect[b, idx, OFF_EFFECT_SOURCE + EffectSource.PENDING] = 1.0
            np_entity_kind[b, idx] = OWNER_PENDING
            idx += 1

        for entity_kind, batch_rows in sources:
            for offset, source, effect in batch_rows[b]:
                if idx == MAX_EFFECTS_STATE:
                    if "effects" not in _WARNED_OVERFLOW:
                        _WARNED_OVERFLOW.add("effects")
                        warnings.warn(
                            f"effect rows exceeded {MAX_EFFECTS_STATE} in one state; "
                            "deep card piles dropped (constants.MAX_EFFECTS_STATE)"
                        )
                    break

                encode_effect_row_into_w_cache(effect, np_effect[b, idx])
                np_effect[b, idx, OFF_EFFECT_SOURCE + source] = 1.0
                np_entity_kind[b, idx] = entity_kind
                np_offset[b, idx] = offset
                idx += 1

        idx_used = max(idx_used, idx)

    # The batch's used rows only: the transport pads back to MAX_EFFECTS_STATE
    np_effect = np.ascontiguousarray(np_effect[:, :idx_used])
    normalize_effect_rows(np_effect)
    return X_Effects(
        x=torch.from_numpy(np_effect).to(device),
        x_entity_kind=torch.from_numpy(np.ascontiguousarray(np_entity_kind[:, :idx_used])).to(
            device
        ),
        x_offset=torch.from_numpy(np.ascontiguousarray(np_offset[:, :idx_used])).to(device),
        batch_size=[batch_size],
    )
