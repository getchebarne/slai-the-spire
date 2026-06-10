import slai
from slai import ActionType


_PENALTY = -0.0010
_WEIGHT_HEALTH_CHAR = 0.0250
_WEIGHT_FLOOR = 0.1000
_WEIGHT_UPGRADE = 0.5000  # Aprox. the reward you'd get from a full value rest


def compute_reward(
    game_state: slai.GameState,
    game_state_next: slai.GameState,
    game_over_flag: bool,
    action: slai.Action,
    gamma: float,
) -> float:
    # Health/floor are potential-based shaping in the Ng et al. form γ·Φ(s') − Φ(s):
    # the discounted sum then telescopes to a constant, whereas plain deltas leak
    # (1−γ)·Φ per step — paying the agent for holding HP/floor instead of winning.
    # γ must be the trainer's discount (threaded from config, not a second constant).
    floor = game_state.map.y_current or 0
    floor_next = game_state_next.map.y_current or 0
    shaped = (
        _WEIGHT_HEALTH_CHAR
        * (gamma * game_state_next.character.health - game_state.character.health)
        + _WEIGHT_FLOOR * (gamma * floor_next - floor)
        # Keyed to the upgrade pick itself (rest-site or event halt): deck-count
        # deltas misfire on purge/transform (−0.5) and duplicate (+0.5) of
        # upgraded cards.
        + _WEIGHT_UPGRADE * float(action.action_type == ActionType.CardUpgrade)
        + _PENALTY
    )

    if not game_over_flag:
        return shaped

    # Terminal: keep the shaped terms — the killing blow's HP loss counts (the
    # health potential anchors at Φ=0 on death) — and add the outcome on top.
    if game_state_next.character.health <= 0:
        return shaped - 1
    return shaped + 1 + game_state_next.character.health / game_state_next.character.health_max
