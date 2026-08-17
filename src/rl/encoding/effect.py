from typing import get_args

import numpy as np
from slai import AmountAbsolute
from slai import AmountRange
from slai import AmountRelative
from slai import CandidatePoolMonsters
from slai import CardKind
from slai import CardPile
from slai import DeltaSign
from slai import Effect
from slai import EffectAdventurerSearch
from slai import EffectBlockGain
from slai import EffectBonfireOffer
from slai import EffectCardAdd
from slai import EffectCardAddRandom
from slai import EffectCardBottle
from slai import EffectCardDiscard
from slai import EffectCardDiscoverPick
from slai import EffectCardDiscoverRoll
from slai import EffectCardDraw
from slai import EffectCardDrawIfNoAttacks
from slai import EffectCardDrawUpTo
from slai import EffectCardDuplicate
from slai import EffectCardExhaust
from slai import EffectCardMove
from slai import EffectCardNightmarePick
from slai import EffectCardPlayFromDrawTop
from slai import EffectCardPurge
from slai import EffectCardRetain
from slai import EffectCardSetupPick
from slai import EffectCardTransform
from slai import EffectCardUpgrade
from slai import EffectCombatEnd
from slai import EffectCombatStart
from slai import EffectDamageFinisher
from slai import EffectDamageFlechettes
from slai import EffectDamageMindBlast
from slai import EffectDamagePhysical
from slai import EffectDamagePhysicalIfPoisoned
from slai import EffectDistractionAdd
from slai import EffectEnergyDelta
from slai import EffectEscapePlanCheck
from slai import EffectEventAdvanceState
from slai import EffectEventConsume
from slai import EffectGamble
from slai import EffectGlassKnifeDecay
from slai import EffectGoldDelta
from slai import EffectHandOfGreedProc
from slai import EffectHealthDelta
from slai import EffectHeelHookProc
from slai import EffectJoustBet
from slai import EffectKnowingSkullAsk
from slai import EffectMausoleumOpen
from slai import EffectMaxHealthDelta
from slai import EffectModifierGain
from slai import EffectModifierMultiply
from slai import EffectModifierRemove
from slai import EffectMonsterSpawn
from slai import EffectPotionAddRandom
from slai import EffectPotionDiscard
from slai import EffectRelicGrantPool
from slai import EffectRelicGrantRandom
from slai import EffectRelicGrantSpecific
from slai import EffectRelicLose
from slai import EffectRewardRollLibraryCards
from slai import EffectRewardRollNeowCards
from slai import EffectRewardRollPotions
from slai import EffectRitualDaggerProc
from slai import EffectScrapOozeReach
from slai import EffectSetCostOverride
from slai import EffectShuffleDiscardPileIntoDrawPile
from slai import EffectSneakyStrikeProc
from slai import EffectStormOfSteelProc
from slai import EffectStrengthLoseTemp
from slai import EffectUnloadDiscard
from slai import EffectWheelSpin
from slai import KnowingSkullWish
from slai import SelectionKindAll
from slai import SelectionKindInput
from slai import SelectionKindInputUpTo
from slai import SelectionKindRandom
from slai import Target
from slai import members

from src.rl.encoding.modifier import MODIFIER_KIND_TO_IDX
from src.rl.encoding.modifier import STACKS_MAX


_NUM_MODIFIER_KINDS = len(MODIFIER_KIND_TO_IDX)
_CARD_KIND_TO_IDX = {kind: int(kind) for kind in members(CardKind)}
_NUM_SKULL_WISHES = len(members(KnowingSkullWish))

# Normalization caps (observed ranges in parentheses)
_DAMAGE_MAX = 64  # sqrt (flat hits 3-60)
_BLOCK_MAX = 20  # linear (2-14)
_DRAW_MAX = 5  # linear (1-3)
_DRAW_UP_TO_MAX = 10  # linear (hand cap)
_ENERGY_MAX = 4  # linear (1-2)
_DISCARD_MAX = 4  # linear (1-3)
_ADD_HAND_MAX = 5  # linear (Blade Dance+ adds 4)
_HP_MAX = 40  # sqrt (event HP swings)
_MAXHP_MAX = 20  # sqrt
_GOLD_MAX = 300  # sqrt (event gold, max ~275)
_DISCOVER_MAX = 5  # linear
_EVENT_ADVANCE_MAX = 3  # linear, signed
_SCRAP_DMG_MAX = 16  # sqrt (3-12)
_MODMUL_FACTOR_MAX = 3  # linear over (factor - 1)
_COST_MAX = 5  # linear (mirrors the card cost caps)
_GLASS_DELTA_MAX = 4  # linear, signed
_HOG_GOLD_MAX = 40  # sqrt (Hand of Greed 20/25 per kill)
_DAGGER_BUMP_MAX = 5  # linear (Ritual Dagger 3/5)
_REWARD_POTIONS_MAX = 3  # linear (Nest 1, potion events 2-3)
_STR_LOSE_TEMP_MAX = 5  # linear (Mutagenic Strength 3)

# ---- Block offsets (the encoding is the concatenation, in this order) ----
# Damage family
_OFF_DMG_TOTAL = 0  # sum of flat hit amounts
_OFF_DMG_MAX_HIT = 1  # max flat hit
_OFF_DMG_INSTANCES = 2  # count of ALL damage effects (flat + conditional)
_OFF_DMG_AOE = 3  # any damage effect on Monsters+All
_OFF_DMG_IF_POISONED = 4
_OFF_DMG_FINISHER = 5
_OFF_DMG_FLECHETTES = 6
_OFF_DMG_MIND_BLAST = 7
_OFF_DMG_COND_RATE = 8  # max static per-unit amount of the conditional kinds
# Block
_OFF_BLOCK_TOTAL = 9
# Draw
_OFF_DRAW_TOTAL = 10
_OFF_DRAW_UP_TO = 11
_OFF_DRAW_UP_TO_AMOUNT = 12
# Energy
_OFF_ENERGY_TOTAL = 13
# Modifier family: per-kind signed summed stacks, split by beneficiary,
# index-aligned with modifier.py's actor-held layout and STACKS_MAX norm
_OFF_MOD_SELF = 14
_OFF_MOD_ENEMY = _OFF_MOD_SELF + _NUM_MODIFIER_KINDS
_OFF_MOD_AOE = _OFF_MOD_ENEMY + _NUM_MODIFIER_KINDS
_OFF_MOD_RANDOM = _OFF_MOD_AOE + 1
_OFF_MOD_BUFF_SELF = _OFF_MOD_RANDOM + 1  # buff/debuff summaries: an unseen
_OFF_MOD_DEBUFF_SELF = _OFF_MOD_BUFF_SELF + 1  # kind still shares its physics
_OFF_MOD_BUFF_ENEMY = _OFF_MOD_DEBUFF_SELF + 1
_OFF_MOD_DEBUFF_ENEMY = _OFF_MOD_BUFF_ENEMY + 1
# Discard (the count lives in the target's SelectionKind)
_OFF_DISCARD_COUNT = _OFF_MOD_DEBUFF_ENEMY + 1
_OFF_DISCARD_INPUT = _OFF_DISCARD_COUNT + 1  # player picks (synergy-enabling)
_OFF_DISCARD_RANDOM = _OFF_DISCARD_INPUT + 1  # uncontrolled loss
# Add-to-hand
_OFF_ADD_HAND_COUNT = _OFF_DISCARD_RANDOM + 1
_OFF_ADD_HAND_UPGRADED = _OFF_ADD_HAND_COUNT + 1
# Health (gains and losses are separate cells, never netted)
_OFF_HP_GAIN_ABS = _OFF_ADD_HAND_UPGRADED + 1
_OFF_HP_LOSS_ABS = _OFF_HP_GAIN_ABS + 1
_OFF_HP_GAIN_REL = _OFF_HP_LOSS_ABS + 1
_OFF_HP_LOSS_REL = _OFF_HP_GAIN_REL + 1
_OFF_MAXHP_GAIN_ABS = _OFF_HP_LOSS_REL + 1
_OFF_MAXHP_LOSS_ABS = _OFF_MAXHP_GAIN_ABS + 1
_OFF_MAXHP_GAIN_REL = _OFF_MAXHP_LOSS_ABS + 1
_OFF_MAXHP_LOSS_REL = _OFF_MAXHP_GAIN_REL + 1
# Gold (events)
_OFF_GOLD_GAIN = _OFF_MAXHP_LOSS_REL + 1  # Absolute amount or Range midpoint
_OFF_GOLD_LOSS = _OFF_GOLD_GAIN + 1
_OFF_GOLD_RANGE = _OFF_GOLD_LOSS + 1
# Discover (potions)
_OFF_DISCOVER_KIND = _OFF_GOLD_RANGE + 1  # CardKind OHE (all-zero = any kind)
_OFF_DISCOVER_COUNT = _OFF_DISCOVER_KIND + len(_CARD_KIND_TO_IDX)
# Deck edits (events; the Deck{filter} target is implied by the kind)
_OFF_ADD_DECK = _OFF_DISCOVER_COUNT + 1
_OFF_ADD_DECK_UPGRADED = _OFF_ADD_DECK + 1
_OFF_PURGE = _OFF_ADD_DECK_UPGRADED + 1
_OFF_UPGRADE = _OFF_PURGE + 1
_OFF_DUPLICATE = _OFF_UPGRADE + 1
_OFF_TRANSFORM = _OFF_DUPLICATE + 1
# Relic grants (events)
_OFF_RELIC_RANDOM = _OFF_TRANSFORM + 1
_OFF_RELIC_SPECIFIC = _OFF_RELIC_RANDOM + 1
# Event flow
_OFF_EVENT_ADVANCE = _OFF_RELIC_SPECIFIC + 1  # signed stage delta
_OFF_EVENT_CONSUME = _OFF_EVENT_ADVANCE + 1
_OFF_SCRAP_DMG = _OFF_EVENT_CONSUME + 1
_OFF_SCRAP_CHANCE = _OFF_SCRAP_DMG + 1
_OFF_SCRAP_ADVANCE_ON_MISS = _OFF_SCRAP_CHANCE + 1
# Long tail (flag, plus a scalar where the kind has a field)
_OFF_MODMUL = _OFF_SCRAP_ADVANCE_ON_MISS + 1  # NOT in the stacks block: x2 poison != +2 poison
_OFF_MODMUL_FACTOR = _OFF_MODMUL + 1
_OFF_SETCOST = _OFF_MODMUL_FACTOR + 1  # flag needed: the amount IS 0 (Bullet Time)
_OFF_SETCOST_AMOUNT = _OFF_SETCOST + 1
_OFF_GAMBLE = _OFF_SETCOST_AMOUNT + 1  # was CalculatedGamble; engine renamed the kind
_OFF_SHUFFLE_DISCARD = _OFF_GAMBLE + 1
_OFF_DISTRACTION = _OFF_SHUFFLE_DISCARD + 1
_OFF_ESCAPE_PLAN = _OFF_DISTRACTION + 1  # conditional block, kept out of BLOCK_TOTAL
_OFF_ESCAPE_PLAN_BLOCK = _OFF_ESCAPE_PLAN + 1
_OFF_GLASS_KNIFE_DELTA = _OFF_ESCAPE_PLAN_BLOCK + 1  # signed; never 0, no flag needed
_OFF_HEEL_HOOK = _OFF_GLASS_KNIFE_DELTA + 1
_OFF_NIGHTMARE_PICK = _OFF_HEEL_HOOK + 1
_OFF_SETUP_PICK = _OFF_NIGHTMARE_PICK + 1
_OFF_SNEAKY_STRIKE_ENERGY = _OFF_SETUP_PICK + 1  # never 0, no flag needed
_OFF_STORM_OF_STEEL = _OFF_SNEAKY_STRIKE_ENERGY + 1
_OFF_UNLOAD = _OFF_STORM_OF_STEEL + 1
_OFF_POTION_ADD = _OFF_UNLOAD + 1
_OFF_POTION_ADD_LIMITED = _OFF_POTION_ADD + 1
_OFF_MODIFIER_REMOVE = _OFF_POTION_ADD_LIMITED + 1  # totality insurance (unused today)
_OFF_CARD_RETAIN = _OFF_MODIFIER_REMOVE + 1  # totality insurance
# ---- Act II additions ----
_OFF_GOLD_REL = _OFF_CARD_RETAIN + 1  # relative gold loss (Masked Bandits' lose-ALL = 1/1)
_OFF_DMG_LIFESTEAL = _OFF_GOLD_REL + 1  # Bite / Suck
_OFF_SETCOST_RANDOM = _OFF_DMG_LIFESTEAL + 1  # Snecko Oil / Confusion re-roll
_OFF_GAMBLE_CHOOSE = _OFF_SETCOST_RANDOM + 1  # Gamble{choose_discards}
_OFF_ADD_PILE_COUNT = _OFF_GAMBLE_CHOOSE + 1  # CardAdd into draw/discard (Wound injection)
_OFF_ADD_PILE_UPGRADED = _OFF_ADD_PILE_COUNT + 1
_OFF_ADD_RANDOM = _OFF_ADD_PILE_UPGRADED + 1  # CardAddRandom (rolled identity)
_OFF_CARD_BOTTLE = _OFF_ADD_RANDOM + 1
_OFF_EXHAUST_COUNT = _OFF_CARD_BOTTLE + 1
_OFF_EXHAUST_RANDOM = _OFF_EXHAUST_COUNT + 1
_OFF_CARD_MOVE = _OFF_EXHAUST_RANDOM + 1
_OFF_CARD_MOVE_FREE = _OFF_CARD_MOVE + 1  # cost_zero is not None
_OFF_DRAW_IF_NO_ATTACKS = _OFF_CARD_MOVE_FREE + 1  # conditional, kept out of DRAW_TOTAL
_OFF_PLAY_FROM_DRAW_TOP = _OFF_DRAW_IF_NO_ATTACKS + 1
_OFF_HOG_GOLD = _OFF_PLAY_FROM_DRAW_TOP + 1  # Hand of Greed per-kill gold
_OFF_RITUAL_DAGGER_BUMP = _OFF_HOG_GOLD + 1
_OFF_POTION_DISCARD = _OFF_RITUAL_DAGGER_BUMP + 1  # We Meet Again's give-potion
_OFF_REWARD_POTIONS_COUNT = _OFF_POTION_DISCARD + 1
_OFF_REWARD_POTIONS_UNIFORM = _OFF_REWARD_POTIONS_COUNT + 1
_OFF_NEOW_CARDS = _OFF_REWARD_POTIONS_UNIFORM + 1
_OFF_NEOW_COLORLESS = _OFF_NEOW_CARDS + 1
_OFF_NEOW_RARE = _OFF_NEOW_COLORLESS + 1
_OFF_LIBRARY_CARDS = _OFF_NEOW_RARE + 1  # 20-card pick-1 bundle roll
_OFF_RELIC_POOL = _OFF_LIBRARY_CARDS + 1  # Cursed Tome books (pool identity via event kind)
_OFF_RELIC_LOSE = _OFF_RELIC_POOL + 1  # which relic: RELIC_EVENT tokens (N'loth)
_OFF_KNOWING_SKULL_WISH = _OFF_RELIC_LOSE + 1  # wish OHE (3): the three repeatable asks
_OFF_JOUST = _OFF_KNOWING_SKULL_WISH + _NUM_SKULL_WISHES
_OFF_JOUST_ON_OWNER = _OFF_JOUST + 1  # 30%/+250g side vs 70%/+100g side
_OFF_MAUSOLEUM = _OFF_JOUST_ON_OWNER + 1
_OFF_WHEEL_SPIN = _OFF_MAUSOLEUM + 1
_OFF_BONFIRE = _OFF_WHEEL_SPIN + 1
_OFF_ADVENTURER_SEARCH = _OFF_BONFIRE + 1
_OFF_MONSTER_SPAWN = _OFF_ADVENTURER_SEARCH + 1  # identity via event kind / Colosseum stage
_OFF_COMBAT_START = _OFF_MONSTER_SPAWN + 1  # this option starts a fight
_OFF_COMBAT_END = _OFF_COMBAT_START + 1
_OFF_COMBAT_END_ESCAPE = _OFF_COMBAT_END + 1  # flee: combat ends with zero rewards
_OFF_STR_LOSE_TEMP = _OFF_COMBAT_END_ESCAPE + 1  # temporary, NOT in the modifier stacks block

ENCODING_DIM_EFFECTS = _OFF_STR_LOSE_TEMP + 1
assert ENCODING_DIM_EFFECTS == 109 + 2 * _NUM_MODIFIER_KINDS + len(_CARD_KIND_TO_IDX)


# ---- Target helpers ----


def _on_enemy(target: Target | None) -> bool:
    return target is not None and isinstance(target.candidate_pool, CandidatePoolMonsters)


def _is_enemy_aoe(target: Target | None) -> bool:
    return _on_enemy(target) and isinstance(target.selection_kind, SelectionKindAll)


def _is_enemy_random(target: Target | None) -> bool:
    return _on_enemy(target) and isinstance(target.selection_kind, SelectionKindRandom)


# ---- Handler factories (write RAW values / 1.0 flags at base-relative offsets) ----


def _flag(off: int):
    def handler(effect, base: int, out: np.ndarray) -> None:
        out[base + off] = 1.0

    return handler


def _accum(off: int, get):
    """Sum-accumulate get(effect) into the slot (normalized once in _finalize)."""

    def handler(effect, base: int, out: np.ndarray) -> None:
        out[base + off] += get(effect)

    return handler


def _flag_and_max(flag_off: int | None, val_off: int, get):
    def handler(effect, base: int, out: np.ndarray) -> None:
        if flag_off is not None:
            out[base + flag_off] = 1.0
        out[base + val_off] = max(out[base + val_off], get(effect))

    return handler


# ---- Family handlers ----


def _flat_damage(effect, base: int, out: np.ndarray) -> None:
    out[base + _OFF_DMG_TOTAL] += effect.amount
    out[base + _OFF_DMG_MAX_HIT] = max(out[base + _OFF_DMG_MAX_HIT], effect.amount)
    out[base + _OFF_DMG_INSTANCES] += 1
    if effect.lifesteal:
        out[base + _OFF_DMG_LIFESTEAL] = 1.0
    if _is_enemy_aoe(effect.target):
        out[base + _OFF_DMG_AOE] = 1.0


def _conditional_damage(flag_off: int, get_rate):
    # Scaling/conditional per-unit rates are not flat damage: flag + rate slot,
    # excluded from the flat totals (instances still counts the hit).
    def handler(effect, base: int, out: np.ndarray) -> None:
        out[base + _OFF_DMG_INSTANCES] += 1
        out[base + flag_off] = 1.0
        out[base + _OFF_DMG_COND_RATE] = max(out[base + _OFF_DMG_COND_RATE], get_rate(effect))
        if _is_enemy_aoe(effect.target):
            out[base + _OFF_DMG_AOE] = 1.0

    return handler


def _modifier_gain(effect, base: int, out: np.ndarray) -> None:
    enemy = _on_enemy(effect.target)
    side = _OFF_MOD_ENEMY if enemy else _OFF_MOD_SELF
    out[base + side + MODIFIER_KIND_TO_IDX[effect.kind]] += effect.stacks
    if effect.kind.is_buff:
        summary = _OFF_MOD_BUFF_ENEMY if enemy else _OFF_MOD_BUFF_SELF
    else:
        summary = _OFF_MOD_DEBUFF_ENEMY if enemy else _OFF_MOD_DEBUFF_SELF
    out[base + summary] += effect.stacks
    if _is_enemy_aoe(effect.target):
        out[base + _OFF_MOD_AOE] = 1.0
    if _is_enemy_random(effect.target):
        out[base + _OFF_MOD_RANDOM] = 1.0


def _picked_count(count_off: int, input_off: int | None, random_off: int):
    """Shared shape of CardDiscard/CardExhaust: the count lives in the target's
    SelectionKind; Input picks are synergy-enabling, Random ones are losses.
    Factory, invoked once at registry build like the others."""

    def handler(effect, base: int, out: np.ndarray) -> None:
        sk = effect.target.selection_kind if effect.target is not None else None
        if isinstance(sk, (SelectionKindInput, SelectionKindInputUpTo)):
            out[base + count_off] += sk.count
            if input_off is not None:
                out[base + input_off] = 1.0
        elif isinstance(sk, SelectionKindRandom):
            out[base + count_off] += sk.count
            out[base + random_off] = 1.0
        else:  # defensive: a Single/All pick counts one
            out[base + count_off] += 1

    return handler


def _card_add(effect, base: int, out: np.ndarray) -> None:
    # One kind, three destinations: hand (combat tempo), deck (permanent), or a
    # combat pile (Wound/Slimed injection). Identity (card_name) is unencoded,
    # matching the old CardAddTo* fidelity.
    pile = effect.pile
    if pile == CardPile.Hand:
        out[base + _OFF_ADD_HAND_COUNT] += effect.count
        out[base + _OFF_ADD_HAND_UPGRADED] = max(
            out[base + _OFF_ADD_HAND_UPGRADED], float(effect.upgraded)
        )
    elif pile == CardPile.Deck:
        out[base + _OFF_ADD_DECK] = 1.0
        out[base + _OFF_ADD_DECK_UPGRADED] = max(
            out[base + _OFF_ADD_DECK_UPGRADED], float(effect.upgraded)
        )
    else:  # Draw / Discard
        out[base + _OFF_ADD_PILE_COUNT] += effect.count
        out[base + _OFF_ADD_PILE_UPGRADED] = max(
            out[base + _OFF_ADD_PILE_UPGRADED], float(effect.upgraded)
        )


def _card_add_random(effect, base: int, out: np.ndarray) -> None:
    out[base + _OFF_ADD_RANDOM] = 1.0
    _card_add(effect, base, out)


def _energy_delta(effect, base: int, out: np.ndarray) -> None:
    sign = 1.0 if effect.sign == DeltaSign.Gain else -1.0
    out[base + _OFF_ENERGY_TOTAL] += sign * effect.amount


def _gamble(effect, base: int, out: np.ndarray) -> None:
    out[base + _OFF_GAMBLE] = 1.0
    if effect.choose_discards:
        out[base + _OFF_GAMBLE_CHOOSE] = 1.0


def _amount_parts(amount) -> tuple[float, float, bool]:
    """Amount union -> (absolute, relative_fraction, is_range). Range folds to
    its midpoint in the absolute channel."""
    if isinstance(amount, AmountAbsolute):
        return float(amount.amount), 0.0, False
    if isinstance(amount, AmountRelative):
        return 0.0, amount.numerator / amount.denominator, False
    if isinstance(amount, AmountRange):
        return (amount.min + amount.max) / 2, 0.0, True
    raise ValueError(f"Unencoded Amount variant: {type(amount).__name__}")


def _health_delta(off_gain_abs: int, off_loss_abs: int, off_gain_rel: int, off_loss_rel: int):
    def handler(effect, base: int, out: np.ndarray) -> None:
        gain = effect.sign == DeltaSign.Gain
        absolute, relative, _ = _amount_parts(effect.amount)
        out[base + (off_gain_abs if gain else off_loss_abs)] += absolute
        out[base + (off_gain_rel if gain else off_loss_rel)] += relative

    return handler


def _gold_delta(effect, base: int, out: np.ndarray) -> None:
    absolute, relative, is_range = _amount_parts(effect.amount)
    if is_range:
        out[base + _OFF_GOLD_RANGE] = 1.0
    out[base + (_OFF_GOLD_GAIN if effect.sign == DeltaSign.Gain else _OFF_GOLD_LOSS)] += absolute
    out[base + _OFF_GOLD_REL] += relative  # loss-only in content (lose-all = 1/1)


def _discover_roll(effect, base: int, out: np.ndarray) -> None:
    if effect.kind is not None:  # None = any kind (all-zero OHE)
        out[base + _OFF_DISCOVER_KIND + _CARD_KIND_TO_IDX[effect.kind]] = 1.0
    out[base + _OFF_DISCOVER_COUNT] += effect.count


def _scrap_ooze(effect, base: int, out: np.ndarray) -> None:
    out[base + _OFF_SCRAP_DMG] = max(out[base + _OFF_SCRAP_DMG], effect.dmg)
    out[base + _OFF_SCRAP_CHANCE] = max(out[base + _OFF_SCRAP_CHANCE], effect.chance)
    if effect.advance_on_miss:
        out[base + _OFF_SCRAP_ADVANCE_ON_MISS] = 1.0


def _set_cost_override(effect, base: int, out: np.ndarray) -> None:
    out[base + _OFF_SETCOST] = 1.0
    out[base + _OFF_SETCOST_AMOUNT] = max(out[base + _OFF_SETCOST_AMOUNT], effect.amount or 0)
    if effect.random:
        out[base + _OFF_SETCOST_RANDOM] = 1.0


def _combat_end(effect, base: int, out: np.ndarray) -> None:
    out[base + _OFF_COMBAT_END] = 1.0
    if effect.escaped_character:
        out[base + _OFF_COMBAT_END_ESCAPE] = 1.0


def _knowing_skull(effect, base: int, out: np.ndarray) -> None:
    out[base + _OFF_KNOWING_SKULL_WISH + int(effect.wish)] = 1.0


def _joust(effect, base: int, out: np.ndarray) -> None:
    out[base + _OFF_JOUST] = 1.0
    if effect.on_owner:
        out[base + _OFF_JOUST_ON_OWNER] = 1.0


def _reward_roll_potions(effect, base: int, out: np.ndarray) -> None:
    out[base + _OFF_REWARD_POTIONS_COUNT] += effect.count
    if effect.uniform:
        out[base + _OFF_REWARD_POTIONS_UNIFORM] = 1.0


def _neow_cards(effect, base: int, out: np.ndarray) -> None:
    out[base + _OFF_NEOW_CARDS] = 1.0
    if effect.colorless:
        out[base + _OFF_NEOW_COLORLESS] = 1.0
    if effect.rare_only:
        out[base + _OFF_NEOW_RARE] = 1.0


def _noop(effect, base: int, out: np.ndarray) -> None:
    pass


_DISPATCH = {
    # Damage family
    EffectDamagePhysical: _flat_damage,
    EffectDamagePhysicalIfPoisoned: _conditional_damage(_OFF_DMG_IF_POISONED, lambda e: e.amount),
    EffectDamageFinisher: _conditional_damage(_OFF_DMG_FINISHER, lambda e: e.damage),
    EffectDamageFlechettes: _conditional_damage(_OFF_DMG_FLECHETTES, lambda e: e.damage),
    EffectDamageMindBlast: _conditional_damage(_OFF_DMG_MIND_BLAST, lambda e: 0),
    # Block / draw / energy
    EffectBlockGain: _accum(_OFF_BLOCK_TOTAL, lambda e: e.amount),
    EffectCardDraw: _accum(_OFF_DRAW_TOTAL, lambda e: e.count),
    EffectCardDrawUpTo: _flag_and_max(_OFF_DRAW_UP_TO, _OFF_DRAW_UP_TO_AMOUNT, lambda e: e.amount),
    EffectCardDrawIfNoAttacks: _accum(_OFF_DRAW_IF_NO_ATTACKS, lambda e: e.count),
    EffectEnergyDelta: _energy_delta,
    # Modifiers
    EffectModifierGain: _modifier_gain,
    EffectModifierMultiply: _flag_and_max(_OFF_MODMUL, _OFF_MODMUL_FACTOR, lambda e: e.factor - 1),
    EffectModifierRemove: _flag(_OFF_MODIFIER_REMOVE),
    EffectStrengthLoseTemp: _accum(_OFF_STR_LOSE_TEMP, lambda e: e.stacks),
    # Hand economy
    EffectCardDiscard: _picked_count(_OFF_DISCARD_COUNT, _OFF_DISCARD_INPUT, _OFF_DISCARD_RANDOM),
    EffectCardExhaust: _picked_count(_OFF_EXHAUST_COUNT, None, _OFF_EXHAUST_RANDOM),
    EffectCardRetain: _flag(_OFF_CARD_RETAIN),
    EffectCardAdd: _card_add,
    EffectCardAddRandom: _card_add_random,
    EffectCardMove: _flag_and_max(
        _OFF_CARD_MOVE, _OFF_CARD_MOVE_FREE, lambda e: float(e.cost_zero is not None)
    ),
    EffectCardPlayFromDrawTop: _flag(_OFF_PLAY_FROM_DRAW_TOP),
    EffectUnloadDiscard: _flag(_OFF_UNLOAD),
    EffectGamble: _gamble,
    EffectShuffleDiscardPileIntoDrawPile: _flag(_OFF_SHUFFLE_DISCARD),
    # Health / gold
    EffectHealthDelta: _health_delta(
        _OFF_HP_GAIN_ABS, _OFF_HP_LOSS_ABS, _OFF_HP_GAIN_REL, _OFF_HP_LOSS_REL
    ),
    EffectMaxHealthDelta: _health_delta(
        _OFF_MAXHP_GAIN_ABS, _OFF_MAXHP_LOSS_ABS, _OFF_MAXHP_GAIN_REL, _OFF_MAXHP_LOSS_REL
    ),
    EffectGoldDelta: _gold_delta,
    EffectHandOfGreedProc: _flag_and_max(None, _OFF_HOG_GOLD, lambda e: e.gold),
    EffectRitualDaggerProc: _flag_and_max(None, _OFF_RITUAL_DAGGER_BUMP, lambda e: e.bump),
    # Discover
    EffectCardDiscoverRoll: _discover_roll,
    EffectCardDiscoverPick: _noop,  # always paired with a Roll; no marginal info
    # Deck edits / relic grants / potions
    EffectCardPurge: _flag(_OFF_PURGE),
    EffectCardUpgrade: _flag(_OFF_UPGRADE),
    EffectCardDuplicate: _flag(_OFF_DUPLICATE),
    EffectCardTransform: _flag(_OFF_TRANSFORM),
    EffectCardBottle: _flag(_OFF_CARD_BOTTLE),
    EffectRelicGrantRandom: _flag(_OFF_RELIC_RANDOM),
    EffectRelicGrantSpecific: _flag(_OFF_RELIC_SPECIFIC),
    EffectRelicGrantPool: _flag(_OFF_RELIC_POOL),
    EffectRelicLose: _flag(_OFF_RELIC_LOSE),
    EffectPotionAddRandom: _flag_and_max(
        _OFF_POTION_ADD, _OFF_POTION_ADD_LIMITED, lambda e: float(e.limited)
    ),
    EffectPotionDiscard: _flag(_OFF_POTION_DISCARD),
    # Rewards (event-staged)
    EffectRewardRollPotions: _reward_roll_potions,
    EffectRewardRollNeowCards: _neow_cards,
    EffectRewardRollLibraryCards: _flag(_OFF_LIBRARY_CARDS),
    # Event flow
    EffectEventAdvanceState: _accum(_OFF_EVENT_ADVANCE, lambda e: e.delta),
    EffectEventConsume: _flag(_OFF_EVENT_CONSUME),
    EffectScrapOozeReach: _scrap_ooze,
    EffectAdventurerSearch: _flag(_OFF_ADVENTURER_SEARCH),
    EffectKnowingSkullAsk: _knowing_skull,
    EffectJoustBet: _joust,
    EffectMausoleumOpen: _flag(_OFF_MAUSOLEUM),
    EffectWheelSpin: _flag(_OFF_WHEEL_SPIN),
    EffectBonfireOffer: _flag(_OFF_BONFIRE),
    # Combat flow (events / Smoke Bomb)
    EffectCombatStart: _flag(_OFF_COMBAT_START),
    EffectCombatEnd: _combat_end,
    EffectMonsterSpawn: _flag(_OFF_MONSTER_SPAWN),
    # Card-mechanic procs
    EffectSetCostOverride: _set_cost_override,
    EffectEscapePlanCheck: _flag_and_max(
        _OFF_ESCAPE_PLAN, _OFF_ESCAPE_PLAN_BLOCK, lambda e: e.block
    ),
    EffectGlassKnifeDecay: _accum(_OFF_GLASS_KNIFE_DELTA, lambda e: e.delta),
    EffectHeelHookProc: _flag(_OFF_HEEL_HOOK),
    EffectCardNightmarePick: _flag(_OFF_NIGHTMARE_PICK),
    EffectCardSetupPick: _flag(_OFF_SETUP_PICK),
    EffectSneakyStrikeProc: _flag_and_max(None, _OFF_SNEAKY_STRIKE_ENERGY, lambda e: e.energy),
    EffectStormOfSteelProc: _flag(_OFF_STORM_OF_STEEL),  # its `upgraded` mirrors card.upgraded
    EffectDistractionAdd: _flag(_OFF_DISTRACTION),
}

# A new engine kind must be encoded deliberately, never dropped — and never
# silently: the registry goes stale at import time, not mid-rollout.
assert set(_DISPATCH) == set(get_args(Effect)), (
    f"effect registry out of sync with the engine: "
    f"missing={sorted(c.__name__ for c in set(get_args(Effect)) - set(_DISPATCH))} "
    f"stale={sorted(c.__name__ for c in set(_DISPATCH) - set(get_args(Effect)))}"
)

_SQRT = "sqrt"
_LINEAR = "linear"

# (offset, width, cap, mode) — applied once after the fold. Flags and OHEs are
# already in [0, 1] and need no entry.
_NORM_SPECS = (
    (_OFF_DMG_TOTAL, 1, _DAMAGE_MAX, _SQRT),
    (_OFF_DMG_MAX_HIT, 1, _DAMAGE_MAX, _SQRT),
    (_OFF_DMG_INSTANCES, 1, 5, _LINEAR),
    (_OFF_DMG_COND_RATE, 1, _DAMAGE_MAX, _SQRT),
    (_OFF_BLOCK_TOTAL, 1, _BLOCK_MAX, _LINEAR),
    (_OFF_DRAW_TOTAL, 1, _DRAW_MAX, _LINEAR),
    (_OFF_DRAW_UP_TO_AMOUNT, 1, _DRAW_UP_TO_MAX, _LINEAR),
    (_OFF_ENERGY_TOTAL, 1, _ENERGY_MAX, _LINEAR),
    (_OFF_MOD_SELF, 2 * _NUM_MODIFIER_KINDS, STACKS_MAX, _SQRT),
    (_OFF_MOD_BUFF_SELF, 4, STACKS_MAX, _SQRT),
    (_OFF_DISCARD_COUNT, 1, _DISCARD_MAX, _LINEAR),
    (_OFF_ADD_HAND_COUNT, 1, _ADD_HAND_MAX, _LINEAR),
    (_OFF_HP_GAIN_ABS, 2, _HP_MAX, _SQRT),
    (_OFF_HP_GAIN_REL, 2, 1, _LINEAR),
    (_OFF_MAXHP_GAIN_ABS, 2, _MAXHP_MAX, _SQRT),
    (_OFF_MAXHP_GAIN_REL, 2, 1, _LINEAR),
    (_OFF_GOLD_GAIN, 2, _GOLD_MAX, _SQRT),
    (_OFF_DISCOVER_COUNT, 1, _DISCOVER_MAX, _LINEAR),
    (_OFF_EVENT_ADVANCE, 1, _EVENT_ADVANCE_MAX, _LINEAR),
    (_OFF_SCRAP_DMG, 1, _SCRAP_DMG_MAX, _SQRT),
    (_OFF_SCRAP_CHANCE, 1, 100, _LINEAR),
    (_OFF_MODMUL_FACTOR, 1, _MODMUL_FACTOR_MAX, _LINEAR),
    (_OFF_SETCOST_AMOUNT, 1, _COST_MAX, _LINEAR),
    (_OFF_ESCAPE_PLAN_BLOCK, 1, _BLOCK_MAX, _LINEAR),
    (_OFF_GLASS_KNIFE_DELTA, 1, _GLASS_DELTA_MAX, _LINEAR),
    (_OFF_SNEAKY_STRIKE_ENERGY, 1, _ENERGY_MAX, _LINEAR),
    (_OFF_GOLD_REL, 1, 1, _LINEAR),
    (_OFF_ADD_PILE_COUNT, 1, _ADD_HAND_MAX, _LINEAR),
    (_OFF_EXHAUST_COUNT, 1, _DISCARD_MAX, _LINEAR),
    (_OFF_DRAW_IF_NO_ATTACKS, 1, _DRAW_MAX, _LINEAR),
    (_OFF_HOG_GOLD, 1, _HOG_GOLD_MAX, _SQRT),
    (_OFF_RITUAL_DAGGER_BUMP, 1, _DAGGER_BUMP_MAX, _LINEAR),
    (_OFF_REWARD_POTIONS_COUNT, 1, _REWARD_POTIONS_MAX, _LINEAR),
    (_OFF_STR_LOSE_TEMP, 1, _STR_LOSE_TEMP_MAX, _LINEAR),
)


def _finalize(base: int, out: np.ndarray) -> None:
    for off, width, cap, mode in _NORM_SPECS:
        view = out[base + off : base + off + width]
        if mode == _SQRT:
            # Vectorized get_sqrt_norm: sign-preserving sqrt compression into [-1, 1]
            clamped = np.clip(view, -cap, cap)
            view[:] = np.sign(clamped) * np.sqrt(np.abs(clamped) / cap)
        else:
            view[:] = np.clip(view / cap, -1.0, 1.0)


def encode_effects_into(effects: list[Effect], pos: int, out: np.ndarray) -> int:
    """Fold an effect list into the shared per-kind block layout at `pos`.

    The region must be zeroed. Raises on an effect variant the layout doesn't
    map — a new engine kind must be encoded deliberately, never dropped.
    """
    for effect in effects:
        handler = _DISPATCH.get(type(effect))
        if handler is None:
            raise ValueError(f"Unencoded effect kind: {type(effect).__name__}")
        handler(effect, pos, out)
    _finalize(pos, out)

    return pos + ENCODING_DIM_EFFECTS
