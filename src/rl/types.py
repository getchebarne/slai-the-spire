from enum import Enum

import torch
from slai import ACTION_SPEC_REGISTRY
from slai import Action
from slai import ActionType
from slai import members
from tensordict import TensorDict
from tensordict import tensorclass

from src.rl.index import TOKEN_SIZE
from src.rl.index import Token
from src.rl.index import TokenContext
from src.rl.index import TokenKind


@tensorclass
class TPadded:
    x: torch.Tensor  # (B, S, D) per-item features
    mask: torch.Tensor  # (B, S) True = valid, False = padding


@tensorclass
class TGameState:
    """Encoded game state: one tensor per token kind (tokens = the kind-local slices
    in index.LOCAL_SLICE) + flat context blocks."""

    # Per-kind entity tensors, token layout per src.rl.index
    cards: TPadded
    relics: TPadded
    potions: TPadded
    monsters: TPadded
    event_options: TPadded
    character: torch.Tensor  # flat singleton, always present
    # Flat context
    energy: torch.Tensor
    map_grid: torch.Tensor  # named map_grid (not map) to avoid shadowing TensorDict.map()
    room_node_idx: torch.Tensor  # (B, MAP_WIDTH) long — next-row room node per column (map GNN)
    map_meta: torch.Tensor
    screen: torch.Tensor
    reward_meta: torch.Tensor
    shop_meta: torch.Tensor
    event_meta: torch.Tensor
    # Per-item shop prices, aligned with the SHOP-context tokens (cards/relics/potions)
    shop_card_prices: torch.Tensor
    shop_relic_prices: torch.Tensor
    shop_potion_prices: torch.Tensor


@tensorclass
class TCoreOutput:
    """Core encoder output: refined entity tokens + global context for the heads.

    `tokens` holds every entity token (B, index.NUM_TOKENS, dim_entity) in registry
    order — consumers slice it with index.GLOBAL_SLICE; the learned global token is
    stripped (its content lives in x_global). Context-only segments (relics piles,
    draw/discard/exhaust, ...) reach x_global via attention and are simply never
    sliced for selection. The next-row map rooms are the ROOM token block, refined
    alongside every other entity — RoomSelect slices them like any other selection.
    Shop prices pass through for the pointer price keys.
    """

    x_global: torch.Tensor  # (B, dim_global)
    x_screen: torch.Tensor  # (B, _ENCODING_DIM_SCREEN) raw flats — L1 GLU context
    tokens: TPadded  # (B, NUM_TOKENS, dim_entity) refined entity tokens
    shop_card_prices: torch.Tensor
    shop_relic_prices: torch.Tensor
    shop_potion_prices: torch.Tensor


@tensorclass
class TMask:
    """Batched action masks (row-indexing / `torch.cat` / `.to(device)` come free)."""

    mask_action_type: torch.Tensor  # (B, NUM_ACTION_TYPES) bool — L1: which action kinds are legal
    mask_action_idx: TensorDict  # {str(int(ActionType)): (B, target_size) bool} — L2, deduped; selecting types only
    mask_target_card: torch.Tensor  # (B, MAX_SIZE_HAND, MAX_MONSTERS) bool — L3 card targets
    mask_target_potion: (
        torch.Tensor
    )  # (B, MAX_POTION_SLOTS, MAX_MONSTERS) bool — L3 potion targets


# =============================================================================
# Action-space tables, keyed off slai.ActionType
# =============================================================================
#
# The engine's `recompute_legal_actions` is the source of truth: a halt exposes only
# the pending pick's actions (one ActionType), so it's just a one-legal-kind state.
# L1 (option) is a masked categorical over `ActionType`; L2 (selection) an entity pick over
# a token (one pointer-key net per kind, conditioned on the ActionType); L3 (target) a
# monster pick for CardPlay / PotionUse. `ACTION_TARGET` maps each action to the token it
# selects (RoomSelect targets the ROOM token block produced by the map GNN); the engine's
# action arity drives the idx shape (selection / optional target). The int-indexed `AT_*`
# views feed the hot paths (no FFI-enum dict keys).


_AT_BY_INT = list(members(ActionType))
NUM_ACTION_TYPES = len(_AT_BY_INT)


# L2 selection target per action type; types absent here are terminal (no selection). Keyed
# by ActionType so it's robust to enum reordering. Pending-only kinds (CardSetup..CardRetain)
# are reached via a halt and appear in L1 only when their halt is the sole legal action.
ACTION_TARGET: dict[ActionType, Token] = {
    ActionType.CardPlay: Token(TokenKind.CARD, TokenContext.HAND),
    ActionType.PotionUse: Token(TokenKind.POTION, TokenContext.OWNED),
    ActionType.PotionDiscard: Token(TokenKind.POTION, TokenContext.OWNED),
    ActionType.RoomSelect: Token(TokenKind.ROOM, None),
    ActionType.CardUpgrade: Token(TokenKind.CARD, TokenContext.OWNED),  # rest/pending upgrade
    ActionType.RewardTakeCard: Token(TokenKind.CARD, TokenContext.REWARD),
    ActionType.ShopBuyCard: Token(TokenKind.CARD, TokenContext.SHOP),
    ActionType.ShopBuyRelic: Token(TokenKind.RELIC, TokenContext.SHOP),
    ActionType.ShopBuyPotion: Token(TokenKind.POTION, TokenContext.SHOP),
    ActionType.ShopPurge: Token(TokenKind.CARD, TokenContext.OWNED),
    ActionType.EventOptionSelect: Token(TokenKind.EVENT, None),
    ActionType.CardSetup: Token(TokenKind.CARD, TokenContext.HAND),
    ActionType.CardNightmare: Token(TokenKind.CARD, TokenContext.HAND),
    ActionType.CardDiscover: Token(TokenKind.CARD, TokenContext.DISCOVER),
    ActionType.CardPurge: Token(TokenKind.CARD, TokenContext.OWNED),
    ActionType.CardDuplicate: Token(TokenKind.CARD, TokenContext.OWNED),
    ActionType.CardTransform: Token(TokenKind.CARD, TokenContext.OWNED),
    ActionType.CardDiscard: Token(TokenKind.CARD, TokenContext.HAND),
    ActionType.CardRetain: Token(TokenKind.CARD, TokenContext.HAND),
}

# The engine's action schema is the source of truth for each action's idx shape via its
# arity (min, max args): (0,0) terminal, (1,1) a selection, (1,2) a selection + optional
# monster target. Deriving may-target and asserting ACTION_TARGET against it means an engine
# arg-shape change fails loudly here rather than silently emitting invalid actions.
_AT_ARITY: list = [ACTION_SPEC_REGISTRY[m].arity for m in _AT_BY_INT]  # (min, max) per ActionType
assert len(ACTION_SPEC_REGISTRY) == NUM_ACTION_TYPES, "registry must cover every ActionType"
assert {int(a) for a in ACTION_TARGET} == {
    at for at in range(NUM_ACTION_TYPES) if _AT_ARITY[at] != (0, 0)
}, "ACTION_TARGET must cover exactly the engine's index-taking actions"

# Int-indexed views over ActionType for the hot paths (None target = terminal, no selection).
AT_TARGET: list["Token | None"] = [ACTION_TARGET.get(m) for m in _AT_BY_INT]
AT_MAY_TARGET: list[bool] = [
    a[1] == 2 for a in _AT_ARITY
]  # optional trailing monster (CardPlay / PotionUse)
SELECTING_ACTION_TYPES: list[int] = [
    at for at in range(NUM_ACTION_TYPES) if AT_TARGET[at] is not None
]


def _target_size(target: "Token | None") -> int:
    """Max selectable entities for an action's target: the token's cap, -1 for terminals.
    Drives the L2 mask shapes. (RoomSelect's ROOM token caps at MAP_WIDTH.)"""
    if target is None:
        return -1
    return TOKEN_SIZE[target]


AT_SIZE: list[int] = [_target_size(t) for t in AT_TARGET]


def action_from_actiontype(at_int: int, selection_index: int, target_index: int) -> Action:
    """Build a slai.Action from the resolved ActionType int + picked indices. The idxs shape
    follows from the action: terminal -> []; selection -> [sel]; may-target with a chosen
    monster -> [sel, tgt]."""
    if AT_TARGET[at_int] is None:
        idxs = []
    elif AT_MAY_TARGET[at_int] and target_index >= 0:
        idxs = [selection_index, target_index]
    else:
        idxs = [selection_index]
    return Action(_AT_BY_INT[at_int], idxs)
