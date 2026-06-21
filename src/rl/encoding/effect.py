import numpy as np
from slai import CandidatePool
from slai import CardKind
from slai import DeltaSign
from slai import Effect
from slai import GoldDeltaKind
from slai import HealthDeltaAmount
from slai import SelectionKind
from slai import Target
from slai import members

from src.rl.encoding.modifier import MODIFIER_KIND_TO_IDX
from src.rl.encoding.modifier import STACKS_MAX


_NUM_MODIFIER_KINDS = len(MODIFIER_KIND_TO_IDX)
_CARD_KIND_TO_IDX = {kind: i for i, kind in enumerate(members(CardKind))}

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
_OFF_GOLD_GAIN = _OFF_MAXHP_LOSS_REL + 1  # Fixed amount or Range midpoint
_OFF_GOLD_LOSS = _OFF_GOLD_GAIN + 1
_OFF_GOLD_RANGE = _OFF_GOLD_LOSS + 1
# Discover (potions)
_OFF_DISCOVER_KIND = _OFF_GOLD_RANGE + 1  # CardKind OHE
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
_OFF_CALCULATED_GAMBLE = _OFF_SETCOST_AMOUNT + 1
_OFF_SHUFFLE_DISCARD = _OFF_CALCULATED_GAMBLE + 1
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

ENCODING_DIM_EFFECTS = _OFF_CARD_RETAIN + 1
assert ENCODING_DIM_EFFECTS == 70 + 2 * _NUM_MODIFIER_KINDS + len(_CARD_KIND_TO_IDX)


# ---- Target helpers ----


def _on_enemy(target: Target | None) -> bool:
    return target is not None and isinstance(target.candidate_pool, CandidatePool.Monsters)


def _is_enemy_aoe(target: Target | None) -> bool:
    return _on_enemy(target) and isinstance(target.selection_kind, SelectionKind.All)


def _is_enemy_random(target: Target | None) -> bool:
    return _on_enemy(target) and isinstance(target.selection_kind, SelectionKind.Random)


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


def _card_discard(effect, base: int, out: np.ndarray) -> None:
    sk = effect.target.selection_kind if effect.target is not None else None
    if isinstance(sk, SelectionKind.Input):
        out[base + _OFF_DISCARD_COUNT] += sk.count
        out[base + _OFF_DISCARD_INPUT] = 1.0
    elif isinstance(sk, SelectionKind.Random):
        out[base + _OFF_DISCARD_COUNT] += sk.count
        out[base + _OFF_DISCARD_RANDOM] = 1.0
    else:  # defensive: a Single/All discard counts one pick
        out[base + _OFF_DISCARD_COUNT] += 1


def _add_to_hand(effect, base: int, out: np.ndarray) -> None:
    out[base + _OFF_ADD_HAND_COUNT] += effect.count
    out[base + _OFF_ADD_HAND_UPGRADED] = max(
        out[base + _OFF_ADD_HAND_UPGRADED], float(effect.upgraded)
    )


def _health_delta(off_gain_abs: int, off_loss_abs: int, off_gain_rel: int, off_loss_rel: int):
    def handler(effect, base: int, out: np.ndarray) -> None:
        gain = effect.sign == DeltaSign.Gain
        amount = effect.amount
        if isinstance(amount, HealthDeltaAmount.Absolute):
            out[base + (off_gain_abs if gain else off_loss_abs)] += amount.amount
        else:  # Relative {numerator, denominator}: a fraction of the base
            out[base + (off_gain_rel if gain else off_loss_rel)] += (
                amount.numerator / amount.denominator
            )

    return handler


def _gold_delta(effect, base: int, out: np.ndarray) -> None:
    if isinstance(effect.kind, GoldDeltaKind.Fixed):
        amount = effect.kind.amount
    else:  # Range {min, max} -> midpoint + flag
        amount = (effect.kind.min + effect.kind.max) / 2
        out[base + _OFF_GOLD_RANGE] = 1.0
    out[base + (_OFF_GOLD_GAIN if effect.sign == DeltaSign.Gain else _OFF_GOLD_LOSS)] += amount


def _discover_roll(effect, base: int, out: np.ndarray) -> None:
    out[base + _OFF_DISCOVER_KIND + _CARD_KIND_TO_IDX[effect.kind]] = 1.0
    out[base + _OFF_DISCOVER_COUNT] += effect.count


def _scrap_ooze(effect, base: int, out: np.ndarray) -> None:
    out[base + _OFF_SCRAP_DMG] = max(out[base + _OFF_SCRAP_DMG], effect.dmg)
    out[base + _OFF_SCRAP_CHANCE] = max(out[base + _OFF_SCRAP_CHANCE], effect.chance)
    if effect.advance_on_miss:
        out[base + _OFF_SCRAP_ADVANCE_ON_MISS] = 1.0


def _noop(effect, base: int, out: np.ndarray) -> None:
    pass


_DISPATCH = {
    # Damage family
    Effect.DamagePhysical: _flat_damage,
    Effect.DamagePhysicalIfPoisoned: _conditional_damage(_OFF_DMG_IF_POISONED, lambda e: e.amount),
    Effect.DamageFinisher: _conditional_damage(_OFF_DMG_FINISHER, lambda e: e.damage),
    Effect.DamageFlechettes: _conditional_damage(_OFF_DMG_FLECHETTES, lambda e: e.damage),
    Effect.DamageMindBlast: _conditional_damage(_OFF_DMG_MIND_BLAST, lambda e: 0),
    # Block / draw / energy
    Effect.BlockGain: _accum(_OFF_BLOCK_TOTAL, lambda e: e.amount),
    Effect.CardDraw: _accum(_OFF_DRAW_TOTAL, lambda e: e.count),
    Effect.CardDrawUpTo: _flag_and_max(
        _OFF_DRAW_UP_TO, _OFF_DRAW_UP_TO_AMOUNT, lambda e: e.amount
    ),
    Effect.EnergyGain: _accum(_OFF_ENERGY_TOTAL, lambda e: e.amount),
    # Modifiers
    Effect.ModifierGain: _modifier_gain,
    Effect.ModifierMultiply: _flag_and_max(
        _OFF_MODMUL, _OFF_MODMUL_FACTOR, lambda e: e.factor - 1
    ),
    Effect.ModifierRemove: _flag(_OFF_MODIFIER_REMOVE),
    # Hand economy
    Effect.CardDiscard: _card_discard,
    Effect.CardRetain: _flag(_OFF_CARD_RETAIN),
    Effect.CardAddToHand: _add_to_hand,
    Effect.UnloadDiscard: _flag(_OFF_UNLOAD),
    Effect.CalculatedGamble: _flag(_OFF_CALCULATED_GAMBLE),
    Effect.ShuffleDiscardPileIntoDrawPile: _flag(_OFF_SHUFFLE_DISCARD),
    # Health / gold
    Effect.HealthDelta: _health_delta(
        _OFF_HP_GAIN_ABS, _OFF_HP_LOSS_ABS, _OFF_HP_GAIN_REL, _OFF_HP_LOSS_REL
    ),
    Effect.MaxHealthDelta: _health_delta(
        _OFF_MAXHP_GAIN_ABS, _OFF_MAXHP_LOSS_ABS, _OFF_MAXHP_GAIN_REL, _OFF_MAXHP_LOSS_REL
    ),
    Effect.GoldDelta: _gold_delta,
    # Discover
    Effect.CardDiscoverRoll: _discover_roll,
    Effect.CardDiscoverPick: _noop,  # always paired with a Roll; no marginal info
    # Deck edits / relic grants / potions
    Effect.CardAddToDeck: _flag_and_max(
        _OFF_ADD_DECK, _OFF_ADD_DECK_UPGRADED, lambda e: float(e.upgraded)
    ),
    Effect.CardPurge: _flag(_OFF_PURGE),
    Effect.CardUpgrade: _flag(_OFF_UPGRADE),
    Effect.CardDuplicate: _flag(_OFF_DUPLICATE),
    Effect.CardTransform: _flag(_OFF_TRANSFORM),
    Effect.RelicGrantRandom: _flag(_OFF_RELIC_RANDOM),
    Effect.RelicGrantSpecific: _flag(_OFF_RELIC_SPECIFIC),
    Effect.PotionAddRandom: _flag_and_max(
        _OFF_POTION_ADD, _OFF_POTION_ADD_LIMITED, lambda e: float(e.limited)
    ),
    # Event flow
    Effect.EventAdvanceState: _accum(_OFF_EVENT_ADVANCE, lambda e: e.delta),
    Effect.EventConsume: _flag(_OFF_EVENT_CONSUME),
    Effect.ScrapOozeReach: _scrap_ooze,
    # Card-mechanic procs
    Effect.SetCostOverride: _flag_and_max(_OFF_SETCOST, _OFF_SETCOST_AMOUNT, lambda e: e.amount),
    Effect.EscapePlanCheck: _flag_and_max(
        _OFF_ESCAPE_PLAN, _OFF_ESCAPE_PLAN_BLOCK, lambda e: e.block
    ),
    Effect.GlassKnifeDecay: _accum(_OFF_GLASS_KNIFE_DELTA, lambda e: e.delta),
    Effect.HeelHookProc: _flag(_OFF_HEEL_HOOK),
    Effect.CardNightmarePick: _flag(_OFF_NIGHTMARE_PICK),
    Effect.CardSetupPick: _flag(_OFF_SETUP_PICK),
    Effect.SneakyStrikeProc: _flag_and_max(None, _OFF_SNEAKY_STRIKE_ENERGY, lambda e: e.energy),
    Effect.StormOfSteelProc: _flag(_OFF_STORM_OF_STEEL),  # its `upgraded` mirrors card.upgraded
    Effect.DistractionAdd: _flag(_OFF_DISTRACTION),
}

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
