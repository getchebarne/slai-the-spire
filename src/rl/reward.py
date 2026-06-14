import numpy as np
import slai


_WEIGHT_HEALTH_CHAR = 0.1000  # 4x (advantage-SNR fix); a policy-invariant potential rescale
_WEIGHT_FLOOR = 0.1000

# The reward is decomposed into streams fitted by separate critic outputs
# (AlphaStar-style value decomposition): the dense shaping streams are
# near-deterministic while the terminal outcome is sparse and high-variance,
# and a single scalar critic fitting their sum mixes the two. GAE is linear in
# rewards, so summing the per-stream advantages reproduces the single-critic
# policy gradient exactly while each baseline fits its own stream.
REWARD_STREAMS = ("outcome", "hp", "progress")


def compute_reward(
    game_state: slai.GameState,
    game_state_next: slai.GameState,
    game_over_flag: bool,
    gamma: float,
) -> np.ndarray:
    """Per-stream reward, index-aligned with REWARD_STREAMS; the total reward is
    the sum over streams."""
    # Health/floor are potential-based shaping in the Ng et al. form γ·Φ(s') − Φ(s):
    # the discounted sum then telescopes to a constant, whereas plain deltas leak
    # (1−γ)·Φ per step — paying the agent for holding HP/floor instead of winning.
    # γ must be the trainer's discount (threaded from config, not a second constant).
    floor = game_state.map.y_current or 0
    floor_next = game_state_next.map.y_current or 0
    hp = _WEIGHT_HEALTH_CHAR * (
        gamma * game_state_next.character.health - game_state.character.health
    )
    # Floor-advancement potential shaping only — no discrete action bonuses and no
    # per-step penalty (minimal shaping: deck-edit/economy value is learned from the
    # outcome + HP streams, not hand-crafted bias).
    progress = _WEIGHT_FLOOR * (gamma * floor_next - floor)

    # Terminal: the shaped streams keep their terms — the killing blow's HP loss
    # counts (the health potential anchors at Φ=0 on death) — and the outcome
    # lands in its own stream.
    outcome = 0.0
    if game_over_flag:
        if game_state_next.character.health <= 0:
            outcome = -1.0
        else:
            outcome = 1.0 + game_state_next.character.health / game_state_next.character.health_max

    return np.array([outcome, hp, progress])
