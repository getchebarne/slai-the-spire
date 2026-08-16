import numpy as np
import slai

_WEIGHT_HEALTH_CHAR = 0.1000  # Φ scale, per HP (4x advantage-SNR fix); via the terminal
# potential it is also the win-quality weight against the ±1 outcome stream.

# Floor-progress reward scale, per floor climbed. Exchange rates it encodes (see
# scripts/verify_reward.py, which asserts them): one floor ≈ 3 HP at mid-range Φ'
# (~0.064/HP), ~5-10x the per-floor time cost, and the full 15-floor climb (~2.3
# discounted) stays below the win margin (~4.6) — it orders DEATHS by depth, where
# win/death provides no gradient (the A20 desert, NEWERA-XV/XVI).
_WEIGHT_FLOOR = 0.2

# The reward is decomposed into streams fitted by separate critic outputs
# (AlphaStar-style value decomposition): the dense shaping streams are
# near-deterministic while the terminal outcome is sparse and high-variance,
# and a single scalar critic fitting their sum mixes the two. GAE is linear in
# rewards, so summing the per-stream advantages reproduces the single-critic
# policy gradient exactly while each baseline fits its own stream.
REWARD_STREAMS = ("outcome", "hp", "floor")


def _potential(health: float, health_max: float) -> float:
    """Health potential Φ(h) = W·(2h − h²/h_max), quadratic in missing HP: losing a
    point of HP costs ~2× the linear rate near death and ~0 near full HP (slope
    2W·(1 − h/h_max)). Φ(0) = 0 keeps the death anchor and Φ(h_max) = W·h_max keeps
    the stream's total swing at the scale W was calibrated on."""
    return _WEIGHT_HEALTH_CHAR * (2.0 * health - health * health / health_max)


def compute_reward(
    game_state: slai.GameState,
    game_state_next: slai.GameState,
    game_over_flag: bool,
    gamma: float,
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
    outcome = -(1.0 - gamma)
    if game_over_flag:
        outcome += -1.0 if game_state_next.character.health <= 0 else 1.0

    # Floor-progress EVENT reward W·Δfloor (not potential form): credit is dated to the
    # climb and never decays. The potential form charged (1−γ)·W·f "rent" per step held,
    # which made dying fast return-optimal in hopeless boss fights — XVII learned to
    # end-turn itself to death at the boss (2026-07-05). Floors are monotone within the
    # act, so the stream can't be farmed; anti-stall is the time cost's job.
    floor = _WEIGHT_FLOOR * (
        float(game_state_next.map.y_current or 0) - float(game_state.map.y_current or 0)
    )

    return np.array([outcome, hp, floor])
