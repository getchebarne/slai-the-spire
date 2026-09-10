import numpy as np
import slai

from src.rl.constants import MAP_HEIGHT


_WEIGHT_HEALTH_CHAR = 0.1000  # Φ scale, per HP (4x advantage-SNR fix); via the terminal
# potential it is also the win-quality weight against the ±1 outcome stream.

# Win pays 3x the death magnitude (EFFECT-IIIb): the act-2 boss check is where
# potions, gold, and deck quality cash out, and at ±1 the win was too quiet
# against the dense shaping streams — 5/8 traced runs died AT the boss with
# full gold and unused belts. Deaths stay −1 at every depth (I1: a targeted
# boss-loss penalty would teach avoiding the boss room)
_WEIGHT_WIN = 3.0

# Floor-progress reward scale, per floor climbed — CUMULATIVE across acts (31
# increments: 15 rows + boss sentinel per act + the seam step). Exchange rates
# at γ=0.999 (see scripts/verify_reward.py, which asserts them): one floor
# ≈ 1.9 HP at mid-range Φ' (~0.064/HP), ~9x the per-floor time cost
# (13·(1−γ)), and the full 31-floor climb (~3.0 discounted) stays under the win
# margin (~4.9) — it orders DEATHS by depth, where win/death provides no
# gradient (the A20 desert, NEWERA-XV/XVI). Raised 0.1 → 0.12 for EFFECT-III:
# EFFECT-II's ~6.5k dip showed the "die shallow and healthy" strategy priced
# within noise of pushing act 2. 0.12 is the ceiling under the I2 win-margin
# invariant — a bigger bump needs a bigger win bonus first. The weight is tied
# to γ: at γ=0.998 no value satisfies both invariants over 31 floors (the
# climb outgrows the discounted win margin before it clears the time cost).
_WEIGHT_FLOOR = 0.12

# Gold stream scale, per gold GAINED. Anchor: one floor of progress ≈ one ~75g
# shop card (0.12 / 75). Gains-only, so spending is never taxed — the stream
# exists to break the RewardTakeGold near-tie (EFFECT-II left 41% of combat
# gold unclaimed), not to make hoarding an objective
_WEIGHT_GOLD = _WEIGHT_FLOOR / 75

# Discarding a Potion while a belt slot is free forfeits a ~50g shop asset
# (EFFECT-II discarded 12+ holdable potions across 6 traced games, one at 9 HP)
_PENALTY_POTION_WASTE = 50 * _WEIGHT_GOLD

# Spending completes the acquire→spend loop. OpenAI Five's shaped-reward table
# prices Gold Spent at exactly 1/10 of Gold Gained ("per unit of gold spent on
# items"; Table 6, arXiv:1912.06680) — gains are never clawed back on spend.
# Scoped to shop actions so gold LOST (Looter/Mugger steals, event forfeits)
# never pays
_WEIGHT_GOLD_SPENT = _WEIGHT_GOLD / 10

_SHOP_SPEND_ACTIONS = frozenset(
    {
        slai.ActionType.ShopBuyCard,
        slai.ActionType.ShopBuyPotion,
        slai.ActionType.ShopBuyRelic,
        slai.ActionType.ShopPurge,
    }
)

_ACT_SPAN = MAP_HEIGHT + 1  # rows 0..14 + the boss-room sentinel (y_current = 15)

# The reward is decomposed into streams fitted by separate critic outputs
# (AlphaStar-style value decomposition): the dense shaping streams are
# near-deterministic while the terminal outcome is sparse and high-variance,
# and a single scalar critic fitting their sum mixes the two. GAE is linear in
# rewards, so summing the per-stream advantages reproduces the single-critic
# policy gradient exactly while each baseline fits its own stream.
REWARD_STREAMS = ("outcome", "hp", "floor", "gold")


def cumulative_floor(game_state: slai.GameState) -> int:
    """Run-cumulative floor, 0..31 over two acts: (act−1)·16 + (y_current or 0).

    y_current=None covers both the act-1 start (floor 0) and the fresh-map
    settle right after the engine's ActTransition (floor 16), so the act seam
    pays exactly one floor: (act=1, y=15) → (act=2, y=None) is 15 → 16.
    Monotone non-decreasing over a run — the event-form stream can't be farmed.
    """
    return (game_state.act - 1) * _ACT_SPAN + (game_state.map.y_current or 0)


def _potential(health: float, health_max: float) -> float:
    """Health potential Φ(h) = W·(2h − h²/h_max), quadratic in missing HP: losing a
    point of HP costs ~2× the linear rate near death and ~0 near full HP (slope
    2W·(1 − h/h_max)). Φ(0) = 0 keeps the death anchor and Φ(h_max) = W·h_max keeps
    the stream's total swing at the scale W was calibrated on."""
    return _WEIGHT_HEALTH_CHAR * (2.0 * health - health * health / health_max)


def shop_gold_spent(
    game_state: slai.GameState, game_state_next: slai.GameState, action: slai.Action | None
) -> int:
    """Gold spent by a shop action this step (0 for any other gold decrease)."""
    if action is None or action.action_type not in _SHOP_SPEND_ACTIONS:
        return 0
    return max(0, game_state.character.gold - game_state_next.character.gold)


def wasted_potion_discard(game_state: slai.GameState, action: slai.Action | None) -> bool:
    """True when `action` discards a Potion while the belt still has room
    (discarding at a full belt stages a swap and is never penalized)."""
    # The FFI belt is dense, so len() is the occupancy (the pre-2026-09 sparse
    # belt made this form dead through EFFECT-IIIf)
    return (
        action is not None
        and action.action_type == slai.ActionType.PotionDiscard
        and len(game_state.potions) < game_state.potion_slots_max
    )


def compute_reward(
    game_state: slai.GameState,
    game_state_next: slai.GameState,
    game_over_flag: bool,
    gamma: float,
    action: slai.Action | None = None,
) -> np.ndarray:
    """Per-stream reward, index-aligned with REWARD_STREAMS; the total reward is
    the sum over streams."""
    # Health is potential-based shaping in the Ng et al. form γ·Φ(s') − Φ(s): the
    # discounted sum then telescopes to the endpoints γ^T·Φ(h_T) − Φ(h_0), whereas
    # plain deltas leak (1−γ)·Φ per step — paying the agent for holding HP instead
    # of winning. Recoverable mid-run dips therefore net ~zero by design.
    # γ must be the trainer's discount (threaded from config, not a second constant).
    hp = gamma * _potential(
        game_state_next.character.health, game_state_next.character.health_max
    ) - _potential(game_state.character.health, game_state.character.health_max)

    # Outcome stream: ±1 on win/death plus a per-step time cost of exactly (1 − γ).
    # The cost's discounted sum over T steps is (1 − γ^T), which cancels the discount
    # on the terminal ±1 — death at ANY T totals ≈ −1, so postponing death earns
    # nothing (NEWERA-XV learned to block-cycle fights forever to dilute the death
    # penalty, 2026-07-05) and slow wins pay a mild time tax. The HP stream keeps its
    # terminal term — the killing blow's HP loss counts and the potential anchors at
    # Φ=0 on death. Win QUALITY lives entirely in the hp endpoint γ^T·Φ(h_T): the
    # terminal potential deliberately does NOT vanish on wins; "fixing" it to
    # Ng-strict invariance would delete the final-HP incentive.
    # Win detection keys on game_over ∧ health>0 ONLY: the Act-II boss kill ends
    # the run with no reward screen (combat_end.rs), and Fairy Potion / Lizard
    # Tail revives return early WITHOUT setting game_over (process_effect_death.rs)
    # — health==0 alone must never be read as a terminal.
    outcome = -(1.0 - gamma)
    if game_over_flag:
        outcome += -1.0 if game_state_next.character.health <= 0 else _WEIGHT_WIN

    # Floor-progress EVENT reward W·Δfloor (not potential form): credit is dated to the
    # climb and never decays. The potential form charged (1−γ)·W·f "rent" per step held,
    # which made dying fast return-optimal in hopeless boss fights — XVII learned to
    # end-turn itself to death at the boss (2026-07-05). The CUMULATIVE floor is
    # monotone across the whole run (the seam pays +1, never −15), so the stream
    # can't be farmed; anti-stall is the time cost's job. A SmokeBomb escape
    # closes the combat with zero rewards but moves no floor — Δ = 0.
    floor = _WEIGHT_FLOOR * float(cumulative_floor(game_state_next) - cumulative_floor(game_state))

    # Gold stream, EVENT form on GAINS only: acquisition pays, spending is free
    # (a symmetric/potential form would tax purchases — the opposite of intent).
    # Gold sources are finite per map, so the stream can't be farmed. The waste
    # penalty prices a discarded-but-holdable Potion as the ~50g asset it was
    gold = _WEIGHT_GOLD * float(max(0, game_state_next.character.gold - game_state.character.gold))
    # All shop spending (purges included) pays the Five 1/10 rate. IIIb paid
    # purges at the GAIN rate; with escalating purge prices that made late
    # purges the highest-paying action in the economy — the agent hoarded gold
    # across shops for them and stopped buying relics entirely (probe
    # 2026-09-01: 17 shops, 12 purges, 0 relics). The purge preference is
    # internalized (it purges at events for zero reward), so IIIc drops the
    # training wheels and lets the outcome stream carry curation
    spent = float(shop_gold_spent(game_state, game_state_next, action))
    gold += _WEIGHT_GOLD_SPENT * spent
    if wasted_potion_discard(game_state, action):
        gold -= _PENALTY_POTION_WASTE

    return np.array([outcome, hp, floor, gold])
