from enum import IntEnum

import torch
from slai import ACTION_SPEC_REGISTRY
from slai import Action
from slai import ActionType
from slai import members
from tensordict import TensorDict
from tensordict import tensorclass

@tensorclass
class TPadded:
    x: torch.Tensor  # (B, S, D) per-item features
    mask: torch.Tensor  # (B, S) True = valid, False = padding


@tensorclass
class TGameState:
    """Encoded game state: one tensor per entity class (segments = the class-local
    slices in index.CLASS_SLICE) + flat context blocks."""

    # Per-class entity tensors, segment layout per src.rl.index
    cards: TPadded
    relics: TPadded
    potions: TPadded
    monsters: TPadded
    event_options: TPadded
    character: torch.Tensor  # flat singleton, always present
    # Flat context
    energy: torch.Tensor
    map_grid: torch.Tensor  # named map_grid (not map) to avoid shadowing TensorDict.map()
    map_meta: torch.Tensor
    screen: torch.Tensor
    reward_meta: torch.Tensor
    shop_meta: torch.Tensor
    event_meta: torch.Tensor
    # Per-item shop prices, aligned with the shop segments (index.POOL_PRICE_FIELD)
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
    sliced for selection. Shop prices pass through for the pointer price keys.
    """

    x_global: torch.Tensor  # (B, dim_global)
    x_screen: torch.Tensor  # (B, _ENCODING_DIM_SCREEN) raw flats — L1 GLU context
    x_map: torch.Tensor  # (B, MAP_WIDTH, dim_map) — per-column embeddings
    tokens: TPadded  # (B, NUM_TOKENS, dim_entity) refined entity tokens
    shop_card_prices: torch.Tensor
    shop_relic_prices: torch.Tensor
    shop_potion_prices: torch.Tensor


@tensorclass
class TMask:
    """Batched action masks (row-indexing / `torch.cat` / `.to(device)` come free)."""

    mask_action_type: torch.Tensor  # (B, NUM_ACTION_TYPES) bool — L1: which action kinds are legal
    mask_action_idx: TensorDict  # {str(int(ActionType)): (B, pool_size) bool} — L2, deduped; selecting types only
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
# L1 (option) is a masked categorical over `ActionType`; L2 (selection) an entity pick
# over a `Pool` (one head per pool, conditioned on the ActionType); L3 (target) a monster
# pick for CardPlay / PotionUse. `ACTION_POOL` maps each action to its RL pool; the engine's
# action arity drives the idx shape (selection / optional target). The int-indexed `AT_*`
# views feed the hot paths (so we never key a dict by an FFI enum at runtime).


_AT_BY_INT = list(members(ActionType))
NUM_ACTION_TYPES = len(_AT_BY_INT)


class Pool(IntEnum):
    HAND = 0
    DECK = 1
    POTIONS = 2
    MAP = 3
    DISCOVER = 4
    REWARD_CARDS = 5
    SHOP_CARDS = 6
    SHOP_RELICS = 7
    SHOP_POTIONS = 8
    EVENT_OPTIONS = 9


# L2 selection pool per action type; types absent here are terminal (no selection). Keyed
# by ActionType so it's robust to enum reordering. Pending-only kinds (CardSetup..CardRetain)
# are reached via a halt and appear in L1 only when their halt is the sole legal action.
ACTION_POOL: dict[ActionType, Pool] = {
    ActionType.CardPlay: Pool.HAND,
    ActionType.PotionUse: Pool.POTIONS,
    ActionType.PotionDiscard: Pool.POTIONS,
    ActionType.RoomSelect: Pool.MAP,
    ActionType.CardUpgrade: Pool.DECK,  # rest-upgrade or pending-upgrade halt
    ActionType.RewardTakeCard: Pool.REWARD_CARDS,
    ActionType.ShopBuyCard: Pool.SHOP_CARDS,
    ActionType.ShopBuyRelic: Pool.SHOP_RELICS,
    ActionType.ShopBuyPotion: Pool.SHOP_POTIONS,
    ActionType.ShopPurge: Pool.DECK,
    ActionType.EventOptionSelect: Pool.EVENT_OPTIONS,
    ActionType.CardSetup: Pool.HAND,
    ActionType.CardNightmare: Pool.HAND,
    ActionType.CardDiscover: Pool.DISCOVER,
    ActionType.CardPurge: Pool.DECK,
    ActionType.CardDuplicate: Pool.DECK,
    ActionType.CardTransform: Pool.DECK,
    ActionType.CardDiscard: Pool.HAND,
    ActionType.CardRetain: Pool.HAND,
}

# The engine's action schema is the source of truth for each action's idx shape via its
# arity (min, max args): (0,0) terminal, (1,1) a selection, (1,2) a selection + optional
# monster target. Deriving may-target and asserting ACTION_POOL against it means an engine
# arg-shape change fails loudly here rather than silently emitting invalid actions.
_AT_ARITY: list = [ACTION_SPEC_REGISTRY[m].arity for m in _AT_BY_INT]  # (min, max) per ActionType
assert len(ACTION_SPEC_REGISTRY) == NUM_ACTION_TYPES, "registry must cover every ActionType"
assert {int(a) for a in ACTION_POOL} == {
    at for at in range(NUM_ACTION_TYPES) if _AT_ARITY[at] != (0, 0)
}, "ACTION_POOL must cover exactly the engine's index-taking actions"

# Int-indexed views over ActionType for the hot paths (-1 pool = terminal, no selection).
AT_POOL: list[int] = [int(ACTION_POOL[m]) if m in ACTION_POOL else -1 for m in _AT_BY_INT]
AT_MAY_TARGET: list[bool] = [
    a[1] == 2 for a in _AT_ARITY
]  # optional trailing monster (CardPlay / PotionUse)
SELECTING_ACTION_TYPES: list[int] = [at for at in range(NUM_ACTION_TYPES) if AT_POOL[at] >= 0]


def action_from_actiontype(at_int: int, selection_index: int, target_index: int) -> Action:
    """Build a slai.Action from the resolved ActionType int + picked indices. The idxs shape
    follows from the action: terminal -> []; selection -> [sel]; may-target with a chosen
    monster -> [sel, tgt]."""
    if AT_POOL[at_int] < 0:
        idxs = []
    elif AT_MAY_TARGET[at_int] and target_index >= 0:
        idxs = [selection_index, target_index]
    else:
        idxs = [selection_index]
    return Action(_AT_BY_INT[at_int], idxs)
