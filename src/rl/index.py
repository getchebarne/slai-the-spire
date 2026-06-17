"""Token layout registry — the single source of truth for entity tokens.

Every entity the model sees is a token identified by `Token(kind, context)`: the KIND is
its entity class (one encoding dim + one shared projection per kind), the CONTEXT is its
zone (hand, shop, reward, owned, ...). Tokens of one kind share an encoding dim and a
projection, so the encoding layer emits ONE tensor per kind (tokens are its kind-local
slices, LOCAL_SLICE) and the projector concatenates the projected kind tensors into ONE
token tensor (tokens are its global slices, GLOBAL_SLICE). REGISTRY order is kind-grouped,
so a kind tensor drops into the token tensor as a contiguous block — no permutation
anywhere encode -> project -> transformer.

Consumers (encoding/, models/, action_space/) import the derived artifacts below;
nothing outside this module enumerates tokens, sizes, or order.
"""

from enum import IntEnum
from typing import NamedTuple

import torch
from slai import GameState

from src.rl.constants import MAP_WIDTH
from src.rl.constants import MAX_EVENT_OPTIONS
from src.rl.constants import MAX_MONSTERS
from src.rl.constants import MAX_POTION_REWARDS
from src.rl.constants import MAX_POTION_SLOTS
from src.rl.constants import MAX_RELIC_REWARDS
from src.rl.constants import MAX_RELICS
from src.rl.constants import MAX_SHOP_POTIONS
from src.rl.constants import MAX_SHOP_RELICS
from src.rl.constants import MAX_SIZE_DECK
from src.rl.constants import MAX_SIZE_DISC_PILE
from src.rl.constants import MAX_SIZE_DISCOVER
from src.rl.constants import MAX_SIZE_DRAW_PILE
from src.rl.constants import MAX_SIZE_EXHAUST
from src.rl.constants import MAX_SIZE_HAND
from src.rl.constants import MAX_SIZE_REWARD_CARDS
from src.rl.constants import MAX_SIZE_SHOP_CARDS

# MAP_WIDTH is re-exported for action_space/models, which size the (non-token) map pool.
__all__ = ["TokenKind", "TokenContext", "Token", "MAP_WIDTH"]


class TokenKind(IntEnum):
    """Entity kind: one encoding dim + one shared projection (and one pointer-key net)
    per kind."""

    CARD = 0
    RELIC = 1
    POTION = 2
    MONSTER = 3
    EVENT = 4
    CHARACTER = 5
    ROOM = 6  # next-row selectable map rooms; encoded model-side by the map GNN


class TokenContext(IntEnum):
    """A card's zone within its kind. REWARD / SHOP / OWNED are shared across kinds (the
    engine reuses the notion); the rest are card-only combat piles. Single-context kinds
    (monster / event / character / room) carry no zone — their tokens use context=None."""

    HAND = 0
    DRAW = 1
    DISCARD = 2
    EXHAUST = 3
    OWNED = 4  # master deck / owned relics / potion belt
    DISCOVER = 5
    REWARD = 6
    SHOP = 7


class Token(NamedTuple):
    """A token's identity: (kind, context). Replaces the old flat Segment enum —
    hashable (dict key) with .kind / .context for the kind/context predicates that
    used to be Pool-membership tests. Single-context kinds use context=None (the kind
    alone identifies them)."""

    kind: TokenKind
    context: TokenContext | None


_K, _C = TokenKind, TokenContext

# (token, token cap), ordered kind -> context. The order matches the legacy Segment order
# so the layout (TYPE_IDX, GLOBAL_SLICE, NUM_TOKENS) is byte-identical. The runtime getter
# — which GameState list each token maps to (None-guards, draw pile's next-to-draw tail) —
# lives in token_entities() below; the single-token kinds have bespoke encoders.
REGISTRY: tuple[tuple[Token, int], ...] = (
    # Cards
    (Token(_K.CARD, _C.HAND), MAX_SIZE_HAND),
    (Token(_K.CARD, _C.DRAW), MAX_SIZE_DRAW_PILE),
    (Token(_K.CARD, _C.DISCARD), MAX_SIZE_DISC_PILE),
    (Token(_K.CARD, _C.EXHAUST), MAX_SIZE_EXHAUST),
    (Token(_K.CARD, _C.OWNED), MAX_SIZE_DECK),  # master deck
    (Token(_K.CARD, _C.DISCOVER), MAX_SIZE_DISCOVER),
    (Token(_K.CARD, _C.REWARD), MAX_SIZE_REWARD_CARDS),
    (Token(_K.CARD, _C.SHOP), MAX_SIZE_SHOP_CARDS),
    # Relics
    (Token(_K.RELIC, _C.OWNED), MAX_RELICS),
    (Token(_K.RELIC, _C.REWARD), MAX_RELIC_REWARDS),
    (Token(_K.RELIC, _C.SHOP), MAX_SHOP_RELICS),
    # Potions (belt slots may hold None mid-list; the class encoder skips them in place)
    (Token(_K.POTION, _C.OWNED), MAX_POTION_SLOTS),
    (Token(_K.POTION, _C.REWARD), MAX_POTION_REWARDS),
    (Token(_K.POTION, _C.SHOP), MAX_SHOP_POTIONS),
    # Single-context kinds (context=None; bespoke encoders: monster needs character
    # health/block and emits incoming damage; event encodes meta + options jointly;
    # character is a flat singleton)
    (Token(_K.MONSTER, None), MAX_MONSTERS),
    (Token(_K.EVENT, None), MAX_EVENT_OPTIONS),
    (Token(_K.CHARACTER, None), 1),
    # Map rooms — produced model-side by the map GNN (not the flat encode path) and
    # injected by Core as the last token block; one slot per next-row column.
    (Token(_K.ROOM, None), MAP_WIDTH),
)

TOKENS: tuple[Token, ...] = tuple(t for t, _ in REGISTRY)
TOKEN_SIZE: dict[Token, int] = {t: s for t, s in REGISTRY}

assert len(TOKEN_SIZE) == len(REGISTRY), "tokens must be unique"
assert [t.kind for t in TOKENS] == sorted(
    (t.kind for t in TOKENS), key=int
), "REGISTRY must be kind-grouped (kind tensors map to contiguous token blocks)"


# =============================================================================
# Runtime entity extraction — the registry's layout read against a live GameState
# =============================================================================


def token_entities(token: Token, gs: GameState) -> list:
    """The entities a token holds in a game state — the registry layout read at runtime.
    Discriminates on kind + context; kept here (not a closure on the registry) so the
    layout stays pure data. Only the token kinds (card/relic/potion) reach this; the
    single-token kinds (monster/event/character) have bespoke encoders and never call it."""
    match token.kind:
        case TokenKind.CARD:
            match token.context:
                case TokenContext.HAND:
                    return gs.hand
                case TokenContext.DRAW:
                    return gs.pile_draw[-MAX_SIZE_DRAW_PILE:]  # next-to-draw tail kept
                case TokenContext.DISCARD:
                    return gs.pile_discard
                case TokenContext.EXHAUST:
                    return gs.pile_exhaust
                case TokenContext.OWNED:
                    return gs.deck
                case TokenContext.DISCOVER:
                    return gs.discover
                case TokenContext.REWARD:
                    return gs.reward.cards if gs.reward is not None else []
                case TokenContext.SHOP:
                    return gs.shop.cards if gs.shop is not None else []
        case TokenKind.RELIC:
            match token.context:
                case TokenContext.OWNED:
                    return gs.relics
                case TokenContext.REWARD:
                    return (
                        [gs.reward.relic]
                        if gs.reward is not None and gs.reward.relic is not None
                        else []
                    )
                case TokenContext.SHOP:
                    return gs.shop.relics if gs.shop is not None else []
        case TokenKind.POTION:
            match token.context:
                case TokenContext.OWNED:
                    return gs.potions
                case TokenContext.REWARD:
                    return (
                        [gs.reward.potion]
                        if gs.reward is not None and gs.reward.potion is not None
                        else []
                    )
                case TokenContext.SHOP:
                    return gs.shop.potions if gs.shop is not None else []
    raise ValueError(f"token_entities has no getter for {token!r}")


# =============================================================================
# Derived artifacts — consumers import these, never re-enumerate tokens
# =============================================================================

# Tokens of each kind (kind-grouped contiguous blocks) + per-kind token counts.
KIND_TOKENS: dict[TokenKind, list[Token]] = {
    kind: [t for t in TOKENS if t.kind == kind] for kind in TokenKind
}
NUM_KIND_TOKENS: dict[TokenKind, int] = {
    kind: sum(TOKEN_SIZE[t] for t in toks) for kind, toks in KIND_TOKENS.items()
}

# Kinds whose tokens come from the flat encode path + EntityProjector. ROOM is the
# exception: its tokens are produced model-side by the map GNN and injected by Core as
# the last block, so the projector covers only these and Core concatenates ROOM after.
PROJECTED_KINDS: list[TokenKind] = [k for k in TokenKind if k is not TokenKind.ROOM]
NUM_PROJECTED_TOKENS: int = sum(TOKEN_SIZE[t] for k in PROJECTED_KINDS for t in KIND_TOKENS[k])

# Local slice of each token within its kind tensor (B, sum(kind sizes), D)
LOCAL_SLICE: dict[Token, slice] = {}
for _toks in KIND_TOKENS.values():
    _offset = 0
    for _t in _toks:
        LOCAL_SLICE[_t] = slice(_offset, _offset + TOKEN_SIZE[_t])
        _offset += TOKEN_SIZE[_t]

# Global slice of each token within the concatenated token tensor (B, NUM_TOKENS, D).
# Kinds are contiguous blocks in registry order, so global = kind offset + local.
GLOBAL_SLICE: dict[Token, slice] = {}
_offset = 0
for _t in TOKENS:
    GLOBAL_SLICE[_t] = slice(_offset, _offset + TOKEN_SIZE[_t])
    _offset += TOKEN_SIZE[_t]

NUM_TOKENS = _offset  # all entity tokens (excludes the model's learned global token)

# Token-slot position -> registry index; drives the type-embedding index buffer.
TYPE_IDX: tuple[int, ...] = tuple(i for i, t in enumerate(TOKENS) for _ in range(TOKEN_SIZE[t]))


def token_counts(mask: torch.Tensor) -> torch.Tensor:
    """Per-token valid-token fractions from a token-tensor mask (B, NUM_TOKENS)
    -> (B, len(TOKENS)), registry order. Cardinality features for the global context
    (deck size, pile sizes, ...) that attention pooling blurs."""
    return torch.cat(
        [
            mask[:, GLOBAL_SLICE[t]].sum(dim=1, keepdim=True).float() / TOKEN_SIZE[t]
            for t in TOKENS
        ],
        dim=1,
    )
