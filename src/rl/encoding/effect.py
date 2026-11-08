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
from typing import Any

import numpy as np
import slai
import torch
from slai import Amount
from slai import AmountRelative
from slai import CostScope
from slai import Effect
from slai import GameState

from src.rl.encoding.buffers import SparseBuffer
from src.rl.encoding.sparse import encode_sparse_into
from src.rl.schema.effect import BUCKET_META
from src.rl.schema.effect import ENCODING_DIM_EFFECT
from src.rl.schema.effect import ENCODING_DIM_SPARSE_EFFECT
from src.rl.schema.effect import IDXS_SCALED
from src.rl.schema.effect import INDEX_CAND_FILTER
from src.rl.schema.effect import INDEX_CAND_POOL
from src.rl.schema.effect import INDEX_CARD_COLOR
from src.rl.schema.effect import INDEX_CARD_KIND
from src.rl.schema.effect import INDEX_CARD_PILE
from src.rl.schema.effect import INDEX_CARD_RARITY
from src.rl.schema.effect import INDEX_COST_SCOPE
from src.rl.schema.effect import INDEX_DELTA_SIGN
from src.rl.schema.effect import INDEX_EFFECT
from src.rl.schema.effect import INDEX_MODIFIER_KIND
from src.rl.schema.effect import INDEX_READ_AT
from src.rl.schema.effect import INDEX_RELIC_EXCLUSION
from src.rl.schema.effect import INDEX_RELIC_TIER
from src.rl.schema.effect import INDEX_REWARD_ROLL_TRIGGER
from src.rl.schema.effect import INDEX_ROUNDING
from src.rl.schema.effect import INDEX_SELECTION_KIND
from src.rl.schema.effect import OFF_ADVANCE_ON_MISS
from src.rl.schema.effect import OFF_AMOUNT_IS_RANGE
from src.rl.schema.effect import OFF_BLOCK
from src.rl.schema.effect import OFF_BLOCK_FROM_HIT
from src.rl.schema.effect import OFF_BOTTOM
from src.rl.schema.effect import OFF_CANDIDATE_FILTER
from src.rl.schema.effect import OFF_CANDIDATE_POOL
from src.rl.schema.effect import OFF_CARD_COLOR
from src.rl.schema.effect import OFF_CARD_KIND
from src.rl.schema.effect import OFF_CARD_NAME
from src.rl.schema.effect import OFF_CARD_PILE
from src.rl.schema.effect import OFF_CARD_RARITY
from src.rl.schema.effect import OFF_CHANCE
from src.rl.schema.effect import OFF_CHOOSE_DISCARDS
from src.rl.schema.effect import OFF_COLORLESS
from src.rl.schema.effect import OFF_COST
from src.rl.schema.effect import OFF_COST_SCOPE
from src.rl.schema.effect import OFF_COST_ZERO_PRESENT
from src.rl.schema.effect import OFF_COUNT
from src.rl.schema.effect import OFF_DAMAGE
from src.rl.schema.effect import OFF_DELTA
from src.rl.schema.effect import OFF_DELTA_SIGN
from src.rl.schema.effect import OFF_EFFECT_KIND
from src.rl.schema.effect import OFF_EFFECT_SOURCE
from src.rl.schema.effect import OFF_ENERGY
from src.rl.schema.effect import OFF_ESCAPED_CHARACTER
from src.rl.schema.effect import OFF_FACTOR
from src.rl.schema.effect import OFF_FREE
from src.rl.schema.effect import OFF_GOLD_ABSOLUTE
from src.rl.schema.effect import OFF_GOLD_RELATIVE
from src.rl.schema.effect import OFF_HEALTH_ABSOLUTE
from src.rl.schema.effect import OFF_HEALTH_RELATIVE
from src.rl.schema.effect import OFF_IF_POISONED
from src.rl.schema.effect import OFF_INSTANCES
from src.rl.schema.effect import OFF_LIFESTEAL
from src.rl.schema.effect import OFF_LIMITED
from src.rl.schema.effect import OFF_MODIFIER_BUFF
from src.rl.schema.effect import OFF_MODIFIER_KIND
from src.rl.schema.effect import OFF_MONSTER_NAME
from src.rl.schema.effect import OFF_ON_OWNER
from src.rl.schema.effect import OFF_ONLY_REDUCE
from src.rl.schema.effect import OFF_RANDOM
from src.rl.schema.effect import OFF_RARE_ONLY
from src.rl.schema.effect import OFF_READ_AT
from src.rl.schema.effect import OFF_RELIC_EXCLUSION
from src.rl.schema.effect import OFF_RELIC_NAME
from src.rl.schema.effect import OFF_RELIC_POOL_SIZE
from src.rl.schema.effect import OFF_RELIC_TIER
from src.rl.schema.effect import OFF_REWARD_ROLL_TRIGGER
from src.rl.schema.effect import OFF_ROUNDING
from src.rl.schema.effect import OFF_SELECT_COUNT
from src.rl.schema.effect import OFF_SELECTION_KIND
from src.rl.schema.effect import OFF_STACKS
from src.rl.schema.effect import OFF_STACKS_NEGATIVE
from src.rl.schema.effect import OFF_TARGETED
from src.rl.schema.effect import OFF_TURNS
from src.rl.schema.effect import OFF_UNIFORM
from src.rl.schema.effect import OFF_UPGRADED
from src.rl.schema.effect import SCALED_CAPS
from src.rl.schema.effect import SCALED_LINEAR
from src.rl.schema.effect import SCALED_SQRT
from src.rl.schema.effect import VOCAB_CARD
from src.rl.schema.effect import VOCAB_MONSTER
from src.rl.schema.effect import VOCAB_RELIC
from src.rl.schema.effect import amount_parts
from src.rl.types import EffectSource


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
    absolute, relative, is_range = amount_parts(amount)
    out[off_abs] += absolute
    out[off_rel] += relative
    if is_range:
        out[OFF_AMOUNT_IS_RANGE] = 1.0

    # A fraction also says which value it reads and how it rounds
    if isinstance(amount, AmountRelative):
        out[OFF_READ_AT + INDEX_READ_AT[amount.read_at]] = 1.0
        out[OFF_ROUNDING + INDEX_ROUNDING[amount.rounding]] = 1.0


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
        out[OFF_COST_SCOPE + INDEX_COST_SCOPE[value]] = 1.0


def _encode_block_gain(out: np.ndarray, effect: slai.EffectBlockGain) -> None:
    out[OFF_BLOCK] += effect.amount


def _encode_bomb_arm(out: np.ndarray, effect: slai.EffectBombArm) -> None:
    out[OFF_DAMAGE] += effect.damage
    out[OFF_TURNS] += effect.turns


def _encode_card_add(out: np.ndarray, effect: slai.EffectCardAdd) -> None:
    out[OFF_CARD_NAME + VOCAB_CARD.get(effect.card_name, 0)] = 1.0
    out[OFF_COUNT] += effect.count
    out[OFF_CARD_PILE + INDEX_CARD_PILE[effect.card_pile]] = 1.0
    if effect.upgraded:
        out[OFF_UPGRADED] = 1.0


def _encode_card_add_random(out: np.ndarray, effect: slai.EffectCardAddRandom) -> None:
    out[OFF_CARD_COLOR + INDEX_CARD_COLOR[effect.color]] = 1.0
    _cost_zero(out, effect.cost_zero)
    out[OFF_COUNT] += effect.count
    if effect.kind is not None:
        out[OFF_CARD_KIND + INDEX_CARD_KIND[effect.kind]] = 1.0
    out[OFF_CARD_PILE + INDEX_CARD_PILE[effect.card_pile]] = 1.0
    if effect.rarity is not None:
        out[OFF_CARD_RARITY + INDEX_CARD_RARITY[effect.rarity]] = 1.0
    if effect.upgraded:
        out[OFF_UPGRADED] = 1.0


def _encode_card_discover_select(out: np.ndarray, effect: slai.EffectCardDiscoverSelect) -> None:
    out[OFF_COUNT] += effect.copies
    _cost_zero(out, effect.cost_zero)
    out[OFF_CARD_PILE + INDEX_CARD_PILE[effect.card_pile]] = 1.0


def _encode_card_discover_roll(out: np.ndarray, effect: slai.EffectCardDiscoverRoll) -> None:
    out[OFF_CARD_COLOR + INDEX_CARD_COLOR[effect.color]] = 1.0
    out[OFF_COUNT] += effect.count
    if effect.kind is not None:
        out[OFF_CARD_KIND + INDEX_CARD_KIND[effect.kind]] = 1.0


def _encode_card_draw(out: np.ndarray, effect: slai.EffectCardDraw) -> None:
    out[OFF_COUNT] += effect.count


def _encode_card_draw_if_no_attacks(
    out: np.ndarray, effect: slai.EffectCardDrawIfNoAttacks
) -> None:
    out[OFF_COUNT] += effect.count


def _encode_card_draw_up_to(out: np.ndarray, effect: slai.EffectCardDrawUpTo) -> None:
    out[OFF_COUNT] += effect.amount


def _encode_card_duplicate(out: np.ndarray, effect: slai.EffectCardDuplicate) -> None:
    out[OFF_CARD_PILE + INDEX_CARD_PILE[effect.card_pile]] = 1.0


def _encode_card_move(out: np.ndarray, effect: slai.EffectCardMove) -> None:
    _cost_zero(out, effect.cost_zero)
    out[OFF_CARD_PILE + INDEX_CARD_PILE[effect.card_pile]] = 1.0


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
    out[OFF_DELTA_SIGN + INDEX_DELTA_SIGN[effect.sign]] = 1.0


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
    out[OFF_DELTA_SIGN + INDEX_DELTA_SIGN[effect.sign]] = 1.0


def _encode_hand_of_greed_proc(out: np.ndarray, effect: slai.EffectHandOfGreedProc) -> None:
    out[OFF_GOLD_ABSOLUTE] += effect.gold


def _encode_health_delta(out: np.ndarray, effect: slai.EffectHealthDelta) -> None:
    _write_amount(out, effect.amount, OFF_HEALTH_ABSOLUTE, OFF_HEALTH_RELATIVE)
    out[OFF_DELTA_SIGN + INDEX_DELTA_SIGN[effect.sign]] = 1.0


def _encode_health_set(out: np.ndarray, effect: slai.EffectHealthSet) -> None:
    _write_amount(out, effect.amount, OFF_HEALTH_ABSOLUTE, OFF_HEALTH_RELATIVE)


def _encode_heel_hook_proc(out: np.ndarray, effect: slai.EffectHeelHookProc) -> None:
    out[OFF_DAMAGE] += effect.damage


def _encode_joust_bet(out: np.ndarray, effect: slai.EffectJoustBet) -> None:
    if effect.on_owner:
        out[OFF_ON_OWNER] = 1.0


def _encode_max_health_delta(out: np.ndarray, effect: slai.EffectMaxHealthDelta) -> None:
    _write_amount(out, effect.amount, OFF_HEALTH_ABSOLUTE, OFF_HEALTH_RELATIVE)
    out[OFF_DELTA_SIGN + INDEX_DELTA_SIGN[effect.sign]] = 1.0


def _encode_modifier_delta(out: np.ndarray, effect: slai.EffectModifierDelta) -> None:
    out[OFF_MODIFIER_KIND + INDEX_MODIFIER_KIND[effect.kind]] = 1.0
    if effect.kind.is_buff:
        out[OFF_MODIFIER_BUFF] = 1.0
    out[OFF_STACKS] += effect.stacks


def _encode_modifier_multiply(out: np.ndarray, effect: slai.EffectModifierMultiply) -> None:
    out[OFF_FACTOR] += effect.factor
    out[OFF_MODIFIER_KIND + INDEX_MODIFIER_KIND[effect.kind]] = 1.0
    if effect.kind.is_buff:
        out[OFF_MODIFIER_BUFF] = 1.0


def _encode_monster_spawn(out: np.ndarray, effect: slai.EffectMonsterSpawn) -> None:
    out[OFF_MONSTER_NAME + VOCAB_MONSTER.get(effect.name, 0)] = 1.0


def _encode_reward_roll_cards(out: np.ndarray, effect: slai.EffectRewardRollCards) -> None:
    out[OFF_COUNT] += effect.bundles
    out[OFF_REWARD_ROLL_TRIGGER + INDEX_REWARD_ROLL_TRIGGER[effect.trigger]] = 1.0


def _encode_monster_split(out: np.ndarray, effect: slai.EffectMonsterSplit) -> None:
    out[OFF_MONSTER_NAME + VOCAB_MONSTER.get(effect.name, 0)] = 1.0


def _encode_monster_summon(out: np.ndarray, effect: slai.EffectMonsterSummon) -> None:
    # Each name's share of the pool
    for name in effect.pool:
        out[OFF_MONSTER_NAME + VOCAB_MONSTER.get(name, 0)] += 1 / len(effect.pool)


def _encode_modifier_remove(out: np.ndarray, effect: slai.EffectModifierRemove) -> None:
    out[OFF_MODIFIER_KIND + INDEX_MODIFIER_KIND[effect.kind]] = 1.0
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
        out[OFF_RELIC_NAME + VOCAB_RELIC.get(name, 0)] = 1.0
    out[OFF_RELIC_POOL_SIZE] += len(effect.pool)


def _encode_relic_grant_random(out: np.ndarray, effect: slai.EffectRelicGrantRandom) -> None:
    out[OFF_RELIC_EXCLUSION + INDEX_RELIC_EXCLUSION[effect.exclusion]] = 1.0
    if effect.tier is not None:
        out[OFF_RELIC_TIER + INDEX_RELIC_TIER[effect.tier]] = 1.0


def _encode_relic_grant_specific(out: np.ndarray, effect: slai.EffectRelicGrantSpecific) -> None:
    out[OFF_RELIC_NAME + VOCAB_RELIC.get(effect.name, 0)] = 1.0


def _encode_reward_roll_gold(out: np.ndarray, effect: slai.EffectRewardRollGold) -> None:
    _write_amount(out, effect.amount, OFF_GOLD_ABSOLUTE, OFF_GOLD_RELATIVE)


def _encode_reward_roll_potions(out: np.ndarray, effect: slai.EffectRewardRollPotions) -> None:
    out[OFF_COUNT] += effect.count
    out[OFF_REWARD_ROLL_TRIGGER + INDEX_REWARD_ROLL_TRIGGER[effect.trigger]] = 1.0


def _encode_reward_roll_relic(out: np.ndarray, effect: slai.EffectRewardRollRelic) -> None:
    for name in effect.pool:
        out[OFF_RELIC_NAME + VOCAB_RELIC.get(name, 0)] = 1.0
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
    out[OFF_COST_SCOPE + INDEX_COST_SCOPE[effect.scope]] = 1.0


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
    out[OFF_EFFECT_KIND + INDEX_EFFECT[type(effect)]] = 1.0

    # Rows the engine never targets carry no `target` field at all
    target = getattr(effect, "target", None)
    if target is not None:
        out[OFF_TARGETED] = 1.0
        out[OFF_CANDIDATE_POOL + INDEX_CAND_POOL[target.candidate_pool]] = 1.0
        out[OFF_SELECTION_KIND + INDEX_SELECTION_KIND[type(target.selection_kind)]] = 1.0
        out[OFF_SELECT_COUNT] = float(getattr(target.selection_kind, "count", 1))

        # Multi-hot over the target's filters; an empty list keeps every candidate
        for candidate_filter in target.filters:
            out[OFF_CANDIDATE_FILTER + INDEX_CAND_FILTER[candidate_filter]] = 1.0

    # Effect Type encoder
    encode = _ENCODERS.get(type(effect))
    if encode is not None:
        encode(out, effect)

    # Buckets read the final raw floats; nothing may write those channels below
    for off_float, off_bucket, cap in BUCKET_META:
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

    rows_gather = rows[..., IDXS_SCALED]
    np.clip(rows_gather, -SCALED_CAPS, SCALED_CAPS, out=rows_gather)
    rows_gather[..., SCALED_LINEAR] /= SCALED_CAPS[SCALED_LINEAR]
    sqrt_vals = rows_gather[..., SCALED_SQRT]
    rows_gather[..., SCALED_SQRT] = np.sign(sqrt_vals) * np.sqrt(
        np.abs(sqrt_vals) / SCALED_CAPS[SCALED_SQRT]
    )
    rows[..., IDXS_SCALED] = rows_gather


def encode_batch_effect_pending_into(
    destination: SparseBuffer,
    slot: int,
    x_lengths: torch.Tensor,
    batch_game_state: list[GameState],
) -> None:
    """Write the pending Effect rows into sparse capacity and their per-state lengths.

    One row per state with a halt pending, every state's after the previous state's.

    Parameters
    ----------
    destination : SparseBuffer
        The reserved pending Effect rows and entries.
    slot : int
        The destination's zero-based slot index.
    x_lengths : torch.Tensor
        (B,) long on the CPU. The selected slot's per-state pending row counts, written in place.
    batch_game_state : list[GameState]
        The states to encode, B of them.
    """
    batch_effect_pending = [game_state.effect_pending for game_state in batch_game_state]
    np_count = x_lengths.numpy()
    np_count[:] = [effect_pending is not None for effect_pending in batch_effect_pending]
    np_effect_rows = np.zeros((np_count.sum(), ENCODING_DIM_EFFECT), dtype=np.float32)
    row = 0
    for effect_pending in batch_effect_pending:
        if effect_pending is not None:
            encode_effect_row_into_w_cache(
                effect_pending, EffectSource.PENDING, np_effect_rows[row]
            )
            row += 1

    normalize_effect_rows(np_effect_rows)
    encode_sparse_into(destination, slot, np_effect_rows, ENCODING_DIM_SPARSE_EFFECT)
