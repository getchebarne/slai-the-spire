import numpy as np
from slai import GameState
from slai import RelicName
from slai import SelectionKindInput
from slai import SelectionKindInputUpTo

from src.rl.schema.focus import BOMB_DAMAGE_MAX
from src.rl.schema.focus import BOMB_TURNS
from src.rl.schema.focus import EFFECT_PENDING_TO_IDX
from src.rl.schema.focus import INPUT_POOL_CAP
from src.rl.schema.focus import OFF_FOCUS_BOMB_DAMAGE_DUE
from src.rl.schema.focus import OFF_FOCUS_CHEST_KIND
from src.rl.schema.focus import OFF_FOCUS_CHEST_OPENED
from src.rl.schema.focus import OFF_FOCUS_DRAW_ORDER_VISIBLE
from src.rl.schema.focus import OFF_FOCUS_KIND
from src.rl.schema.focus import OFF_FOCUS_PENDING_COUNT
from src.rl.schema.focus import OFF_FOCUS_PENDING_KIND
from src.rl.schema.focus import OFF_FOCUS_PENDING_SKIPPABLE
from src.rl.schema.focus import OFF_FOCUS_PRESENCE
from src.rl.schema.focus import OFF_FOCUS_REST_DONE
from src.rl.schema.focus import Focus


def derive_focus(game_state: GameState) -> Focus:
    if game_state.reward is not None:
        return Focus.REWARD
    if game_state.combat is not None:
        return Focus.COMBAT
    if game_state.shop is not None:
        return Focus.SHOP
    if game_state.chest is not None:
        return Focus.CHEST
    if game_state.rest_site is not None:
        return Focus.REST_SITE
    if game_state.event is not None:
        return Focus.EVENT

    return Focus.MAP


def draw_order_visible(game_state: GameState) -> bool:
    """Return whether Frozen Eye shows the draw pile's order.

    Parameters
    ----------
    game_state : GameState
        The state to read.

    Returns
    -------
    bool
        True when the Character holds Frozen Eye.
    """
    return any(relic.name == RelicName.FrozenEye for relic in game_state.relics)




def encode_focus_into(game_state: GameState, out: np.ndarray) -> None:
    combat = game_state.combat
    chest = game_state.chest
    rest_site = game_state.rest_site
    pending = game_state.effect_pending

    out[OFF_FOCUS_KIND + derive_focus(game_state)] = 1.0

    # Pending input
    if pending is not None:
        idx = EFFECT_PENDING_TO_IDX.get(type(pending))
        if idx is None:
            raise ValueError(f"Unsupported pending halt kind: {type(pending).__name__}")

        out[OFF_FOCUS_PENDING_KIND + idx] = 1.0

        target = pending.target
        selection_kind = target.selection_kind if target is not None else None
        if isinstance(selection_kind, (SelectionKindInput, SelectionKindInputUpTo)):
            cap = INPUT_POOL_CAP.get(target.candidate_pool)
            if cap is None:
                raise ValueError(
                    f"Input selection from unexpected pool {target.candidate_pool}"
                    f" (pending {type(pending).__name__})"
                )

            out[OFF_FOCUS_PENDING_COUNT] = selection_kind.count / cap
            if isinstance(selection_kind, SelectionKindInputUpTo):
                out[OFF_FOCUS_PENDING_SKIPPABLE] = 1.0

    if combat is not None:
        for turns_left, damage in combat.bombs:
            out[OFF_FOCUS_BOMB_DAMAGE_DUE + turns_left - 1] += damage
        due = out[OFF_FOCUS_BOMB_DAMAGE_DUE : OFF_FOCUS_BOMB_DAMAGE_DUE + BOMB_TURNS]
        due[:] = np.minimum(due, BOMB_DAMAGE_MAX) / BOMB_DAMAGE_MAX

    if chest is not None:
        out[OFF_FOCUS_CHEST_KIND + int(chest.kind)] = 1.0
        out[OFF_FOCUS_CHEST_OPENED] = float(chest.opened)

    if rest_site is not None:
        out[OFF_FOCUS_REST_DONE] = float(rest_site.done)

    # Context-presence bits: the Focus OHE alone can't express overlays
    for i, ctx in enumerate(
        (combat, game_state.reward, game_state.shop, game_state.event, rest_site, chest)
    ):
        if ctx is not None:
            out[OFF_FOCUS_PRESENCE + i] = 1.0

    out[OFF_FOCUS_DRAW_ORDER_VISIBLE] = float(draw_order_visible(game_state))
