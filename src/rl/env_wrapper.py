"""Per-env wrapper that mediates `slai.GameEnv` to preserve the trainer's
two-step card-targeting protocol.

slai bundles target into a single `Action.CardPlay(idx_hand, idx_monster)`,
but the existing trainer emits two separate model decisions (pick card,
then pick monster). The wrapper buffers the pending hand index when the
chosen card requires a target, re-presents the same observation back to
the trainer (with `is_awaiting_target=True`), and only calls
`env.step(slai.Action.CardPlay(...))` once the trainer also picks a monster.

Design:
- `to_action` (in `action_space.types`) returns either a real `slai.Action.*`
  or one of the two markers below.
- `EnvWrapper.step` interprets the markers and either buffers / re-presents
  the obs, or constructs the bundled CardPlay and forwards to slai.
- Routing (in `action_space.route`) consults `wrapper.is_awaiting_target`
  to override the natural phase routing.
"""

import slai


class _PendingCardPlay:
    """Marker emitted when the trainer picks a card to play. The wrapper
    decides whether to buffer-and-await-target or play immediately."""

    __slots__ = ("idx_hand",)

    def __init__(self, idx_hand: int):
        self.idx_hand = idx_hand


class _ResolveCardPlay:
    """Marker emitted when the trainer picks a target for a buffered card."""

    __slots__ = ("idx_monster",)

    def __init__(self, idx_monster: int):
        self.idx_monster = idx_monster


class EnvWrapper:
    """Wraps `slai.GameEnv` with two-step card-targeting buffering."""

    def __init__(self, ascension: int = 0):
        self._env = slai.GameEnv(ascension=ascension)
        self._obs: slai.GameState | None = None
        self._pending_idx_hand: int | None = None

    def reset(self, seed: int) -> slai.GameState:
        self._obs, _ = self._env.reset(seed=seed)
        self._pending_idx_hand = None
        return self._obs

    @property
    def obs(self) -> slai.GameState:
        assert self._obs is not None, "call reset() first"
        return self._obs

    @property
    def is_awaiting_target(self) -> bool:
        return self._pending_idx_hand is not None

    def step(self, action) -> tuple[slai.GameState, float, bool, bool, dict]:
        """Apply an action.

        `action` may be:
          - `_PendingCardPlay(idx_hand)` — pick a card to play
          - `_ResolveCardPlay(idx_monster)` — pick a target for the buffered card
          - any `slai.Action.*` instance — passed straight through to slai
        """
        if isinstance(action, _PendingCardPlay):
            assert self._pending_idx_hand is None, "double-buffered card play"
            card = self._obs.hand[action.idx_hand]
            if card.requires_target:
                # Buffer, hand control back without stepping the engine.
                self._pending_idx_hand = action.idx_hand
                return self._obs, 0.0, False, False, {"awaiting_target": True}
            action = slai.Action.CardPlay(idx_hand=action.idx_hand, idx_monster=None)
        elif isinstance(action, _ResolveCardPlay):
            assert self._pending_idx_hand is not None, "no buffered card to resolve"
            action = slai.Action.CardPlay(
                idx_hand=self._pending_idx_hand,
                idx_monster=action.idx_monster,
            )
            self._pending_idx_hand = None

        self._obs, reward, terminated, truncated, info = self._env.step(action)
        return self._obs, reward, terminated, truncated, info
