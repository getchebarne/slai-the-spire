from enum import IntEnum

import torch
from slai import ACTION_SPEC_REGISTRY
from slai import Action
from slai import ActionType
from slai import members
from tensordict import TensorDict
from tensordict import tensorclass

from src.rl.constants import MAP_WIDTH
from src.rl.constants import MAX_EVENT_OPTIONS
from src.rl.constants import MAX_POTION_SLOTS
from src.rl.constants import MAX_SHOP_CARDS
from src.rl.constants import MAX_SHOP_POTIONS
from src.rl.constants import MAX_SHOP_RELICS
from src.rl.constants import MAX_SIZE_COMBAT_CARD_REWARD
from src.rl.constants import MAX_SIZE_DECK
from src.rl.constants import MAX_SIZE_DISCOVER
from src.rl.constants import MAX_SIZE_HAND


@tensorclass
class TPadded:
    x: torch.Tensor  # (B, S, D) per-item features
    mask: torch.Tensor  # (B, S) True = valid, False = padding


@tensorclass
class TCombat:
    hand: TPadded
    draw: TPadded
    discard: TPadded
    exhaust: TPadded
    deck: TPadded
    monsters: TPadded
    energy: torch.Tensor
    discover: TPadded


@tensorclass
class TReward:
    cards: TPadded
    relic: TPadded
    potion: TPadded
    meta: torch.Tensor


@tensorclass
class TShop:
    cards: TPadded
    card_prices: torch.Tensor
    relics: TPadded
    relic_prices: torch.Tensor
    potions: TPadded
    potion_prices: torch.Tensor
    meta: torch.Tensor


@tensorclass
class TEvent:
    meta: torch.Tensor
    options: TPadded


@tensorclass
class TGameState:
    # Persistent / cross-screen
    character: torch.Tensor
    relics: TPadded
    potions: TPadded
    map_grid: torch.Tensor  # named map_grid (not map) to avoid shadowing TensorDict.map()
    map_meta: torch.Tensor
    screen: torch.Tensor
    # Per-screen
    combat: TCombat
    reward: TReward
    shop: TShop
    event: TEvent


@tensorclass
class TEntityProjection:
    # Card piles
    hand: TPadded
    draw: TPadded
    discard: TPadded
    exhaust: TPadded
    deck: TPadded
    discover: TPadded

    # Combat actors
    monsters: TPadded
    character: torch.Tensor

    # Reward
    reward_cards: TPadded
    reward_relic: TPadded
    reward_potion: TPadded

    # Owned
    relics: TPadded
    potions: TPadded

    # Shop items
    shop_cards: TPadded
    shop_relics: TPadded
    shop_potions: TPadded

    # Event
    event_options: TPadded


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


# Max entities per pool, indexed by Pool. The only per-pool fact that isn't derivable from
# the pool name: the head's CoreOutput tensor is `x_<pool>` and shop pools carry a price
# (both resolved in actor_critic), and the dedup source pile lives in masks._pile_ids.
POOL_SIZE: list[int] = [
    MAX_SIZE_HAND,  # HAND
    MAX_SIZE_DECK,  # DECK
    MAX_POTION_SLOTS,  # POTIONS
    MAP_WIDTH,  # MAP
    MAX_SIZE_DISCOVER,  # DISCOVER
    MAX_SIZE_COMBAT_CARD_REWARD,  # REWARD_CARDS
    MAX_SHOP_CARDS,  # SHOP_CARDS
    MAX_SHOP_RELICS,  # SHOP_RELICS
    MAX_SHOP_POTIONS,  # SHOP_POTIONS
    MAX_EVENT_OPTIONS,  # EVENT_OPTIONS
]
assert len(POOL_SIZE) == len(Pool), "POOL_SIZE must cover every Pool"


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
