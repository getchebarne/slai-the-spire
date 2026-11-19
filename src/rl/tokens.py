"""The engine collections behind tokens, their caps, and a state's tokens in window order."""

import math
import warnings
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Sequence
from inspect import signature
from typing import Any, get_args

from slai import Card
from slai import Character
from slai import Combat
from slai import Event
from slai import EventOption
from slai import GameState
from slai import Map
from slai import Monster
from slai import Potion
from slai import Relic
from slai import RelicName
from slai import Reward
from slai import Room
from slai import Shop

from src.rl.constants import MAP_HEIGHT
from src.rl.constants import MAP_WIDTH
from src.rl.constants import MAX_CARD_REWARD_ROLL
from src.rl.constants import MAX_COMBAT_CARD_REWARD
from src.rl.constants import MAX_EVENT_OPTIONS
from src.rl.constants import MAX_POTION_REWARDS
from src.rl.constants import MAX_POTION_SLOTS
from src.rl.constants import MAX_RELIC_REWARDS
from src.rl.constants import MAX_RELICS
from src.rl.constants import MAX_SHOP_POTIONS
from src.rl.constants import MAX_SHOP_RELICS
from src.rl.constants import MAX_SIZE_CARD_MATCH_PAIRS
from src.rl.constants import MAX_SIZE_CARD_PILE_DECK
from src.rl.constants import MAX_SIZE_CARD_PILE_DISCARD
from src.rl.constants import MAX_SIZE_CARD_PILE_DISCOVER
from src.rl.constants import MAX_SIZE_CARD_PILE_DRAW
from src.rl.constants import MAX_SIZE_CARD_PILE_EVENT_ROLL
from src.rl.constants import MAX_SIZE_CARD_PILE_EXHAUST
from src.rl.constants import MAX_SIZE_CARD_PILE_HAND
from src.rl.constants import MAX_SIZE_CARD_PILE_NIGHTMARE
from src.rl.constants import MAX_SIZE_CARD_PILE_SELECTED
from src.rl.constants import MAX_SIZE_CARD_PILE_STASIS
from src.rl.constants import MAX_SIZE_CARD_PLAY_QUEUE
from src.rl.constants import MAX_SIZE_POTION_EVENT_ROLL
from src.rl.constants import MAX_SIZE_RELIC_EVENT_ROLL
from src.rl.constants import MAX_SIZE_SHOP_CARDS
from src.rl.constants import TOKEN_KINDS_IN_WINDOW_ORDER
from src.rl.types import Entity
from src.rl.types import Token
from src.rl.types import TokenKind


__all__ = [
    "get_entities_for_token_kind",
    "iter_tokens",
    "relic_held",
    "get_room_entities",
    "get_room_burning",
    "get_burning_path",
]


# The token kinds that have already warned about overflow
_WARNED_OVERFLOW: set[TokenKind] = set()

# The burning path by Map, current Room and Wing Boots charges left
_BURNING_PATH_CACHE: dict[tuple[int | None, ...], frozenset[tuple[int, int]]] = {}
_BURNING_PATH_CACHE_MAX = 100_000


def get_entities_for_token_kind(token_kind: TokenKind, game_state: GameState) -> Sequence[Entity]:
    """Return the Entities of a token kind's tokens in rank order.

    Encoding and action decoding use these same lists, so an Entity's id and its token share
    one placement.

    Parameters
    ----------
    token_kind : TokenKind
        The token kind.
    game_state : GameState
        The state to read.

    Returns
    -------
    Sequence[Entity]
        One Entity per token of the token kind, in rank order.
    """
    read_context, get_entities = _GETTER_OF_TOKEN_KIND[token_kind]
    context = read_context(game_state)
    return get_entities(context) if context is not None else []


def iter_tokens(game_state: GameState) -> Iterator[Token]:
    """Yield a state's tokens in the shared order used by encoding and action decoding.

    Parameters
    ----------
    game_state : GameState
        The state whose capped collections supply the tokens.

    Yields
    ------
    Token
        Each token kind and Entity, in `TOKEN_KINDS_IN_WINDOW_ORDER` and then collection order.
    """
    context_of_read_context = {
        read_context: read_context(game_state) for read_context in _READ_CONTEXTS
    }
    for token_kind, read_context, get_entities in _GETTERS_IN_WINDOW_ORDER:
        context = context_of_read_context[read_context]
        if context is not None:
            for entity in get_entities(context):
                yield token_kind, entity


def _capped(token_kind: TokenKind, cards: list[Card], cap: int) -> list[Card]:
    """Cut `cards` to `cap`, warning once per token kind when that drops any.

    Parameters
    ----------
    token_kind : TokenKind
        The Cards' token kind.
    cards : list[Card]
        The Cards to cut.
    cap : int
        The token kind's cap.

    Returns
    -------
    list[Card]
        The first `cap` Cards.
    """
    if len(cards) > cap and token_kind not in _WARNED_OVERFLOW:
        _WARNED_OVERFLOW.add(token_kind)
        warnings.warn(
            f"{token_kind.name} holds {len(cards)} cards; the ones past its cap of {cap} are "
            "dropped"
        )
    return cards[:cap]


# Each context's reader: the state's Combat, Event, Reward or Shop, None while it is inactive
def _read_combat(game_state: GameState) -> Combat | None:
    return game_state.combat


def _read_event(game_state: GameState) -> Event | None:
    return game_state.event


def _read_reward(game_state: GameState) -> Reward | None:
    return game_state.reward


def _read_shop(game_state: GameState) -> Shop | None:
    return game_state.shop


# The token kinds of no context read the state itself
def _read_game_state(game_state: GameState) -> GameState:
    return game_state


# Each collection truncated to its own cap: a Card past it has no token, so masks cannot name it
def _get_card_match_flipped(event: Event) -> list[Card]:
    card = event.card_match_flipped
    return [card] if card is not None else []


def _get_card_match_pairs(event: Event) -> list[Card]:
    return event.card_match_pairs[:MAX_SIZE_CARD_MATCH_PAIRS]


def _get_card_pile_deck(game_state: GameState) -> list[Card]:
    return _capped(TokenKind.CARD_PILE_DECK, game_state.card_pile_deck, MAX_SIZE_CARD_PILE_DECK)


def _get_card_pile_discard(combat: Combat) -> list[Card]:
    return _capped(
        TokenKind.CARD_PILE_DISCARD, combat.card_pile_discard, MAX_SIZE_CARD_PILE_DISCARD
    )


def _get_card_pile_discover(combat: Combat) -> list[Card]:
    return combat.card_pile_discover[:MAX_SIZE_CARD_PILE_DISCOVER]


def _get_card_pile_draw(combat: Combat) -> list[Card]:
    # Known Cards by depth, then the rest by id: the order hides what the player can't see
    cards_draw = sorted(
        combat.card_pile_draw,
        key=lambda card: (card.draw_depth is None, card.draw_depth or 0, card.id),
    )
    return _capped(TokenKind.CARD_PILE_DRAW, cards_draw, MAX_SIZE_CARD_PILE_DRAW)


def _get_card_pile_event_roll(event: Event) -> list[Card]:
    return event.card_pile_event_roll[:MAX_SIZE_CARD_PILE_EVENT_ROLL]


def _get_card_pile_exhaust(combat: Combat) -> list[Card]:
    return combat.card_pile_exhaust[:MAX_SIZE_CARD_PILE_EXHAUST]


def _get_card_pile_hand(combat: Combat) -> list[Card]:
    return combat.card_pile_hand[:MAX_SIZE_CARD_PILE_HAND]


def _get_card_pile_nightmare(combat: Combat) -> list[Card]:
    return _capped(
        TokenKind.CARD_PILE_NIGHTMARE, combat.card_pile_nightmare, MAX_SIZE_CARD_PILE_NIGHTMARE
    )


def _get_card_pile_selected(game_state: GameState) -> list[Card]:
    return game_state.card_pile_selected[:MAX_SIZE_CARD_PILE_SELECTED]


def _get_card_pile_stasis(combat: Combat) -> list[Card]:
    # Index-aligned with the Monsters: a Monster holding no Card is None
    cards = [card for card in combat.card_pile_stasis if card is not None]
    return cards[:MAX_SIZE_CARD_PILE_STASIS]


def _get_card_play_queue(combat: Combat) -> list[Card]:
    return _capped(TokenKind.CARD_PLAY_QUEUE, combat.card_play_queue, MAX_SIZE_CARD_PLAY_QUEUE)


def _get_card_reward_bundle(reward: Reward, idx_bundle: int, cap: int) -> list[Card]:
    cards = reward.cards
    return cards[idx_bundle][:cap] if idx_bundle < len(cards) else []


def _get_card_reward_0(reward: Reward) -> list[Card]:
    # Bundle 0 must hold The Library's single wide roll
    return _get_card_reward_bundle(reward, 0, MAX_CARD_REWARD_ROLL)


def _get_card_reward_1(reward: Reward) -> list[Card]:
    return _get_card_reward_bundle(reward, 1, MAX_COMBAT_CARD_REWARD)


def _get_card_reward_2(reward: Reward) -> list[Card]:
    return _get_card_reward_bundle(reward, 2, MAX_COMBAT_CARD_REWARD)


def _get_card_reward_3(reward: Reward) -> list[Card]:
    return _get_card_reward_bundle(reward, 3, MAX_COMBAT_CARD_REWARD)


def _get_card_reward_4(reward: Reward) -> list[Card]:
    return _get_card_reward_bundle(reward, 4, MAX_COMBAT_CARD_REWARD)


def _get_card_shop(shop: Shop) -> list[Card]:
    return shop.cards[:MAX_SIZE_SHOP_CARDS]


def _get_character(game_state: GameState) -> list[Character]:
    return [game_state.character]


def _get_event_options(event: Event) -> list[EventOption]:
    return event.options[:MAX_EVENT_OPTIONS]


def _get_monsters(combat: Combat) -> list[Monster]:
    return combat.monsters


# Each collection truncated to its own cap (all engine-bounded, so never lossy)
def _get_potion_event_roll(event: Event) -> list[Potion]:
    return event.potion_event_roll[:MAX_SIZE_POTION_EVENT_ROLL]


def _get_potion_owned(game_state: GameState) -> list[Potion]:
    return game_state.potions[:MAX_POTION_SLOTS]


def _get_potion_reward(reward: Reward) -> list[Potion]:
    return reward.potions[:MAX_POTION_REWARDS]


def _get_potion_shop(shop: Shop) -> list[Potion]:
    return shop.potions[:MAX_SHOP_POTIONS]


# Each collection truncated to its own cap: a Relic past it has no token
def _get_relic_event_roll(event: Event) -> list[Relic]:
    return event.relic_event_roll[:MAX_SIZE_RELIC_EVENT_ROLL]


def _get_relic_owned(game_state: GameState) -> list[Relic]:
    return game_state.relics[:MAX_RELICS]


def _get_relic_reward(reward: Reward) -> list[Relic]:
    return reward.relics[:MAX_RELIC_REWARDS]


def _get_relic_shop(shop: Shop) -> list[Relic]:
    return shop.relics[:MAX_SHOP_RELICS]


def _get_wing_boots_charges(game_state: GameState) -> int:
    """Return the charges left on the Character's Wing Boots.

    With charges left, the engine offers every next-row Room, not just the edges.

    Parameters
    ----------
    game_state : GameState
        The state to read.

    Returns
    -------
    int
        The Wing Boots Relic's counter; 0 without the Relic.
    """
    return next(
        (relic.counter for relic in game_state.relics if relic.name == RelicName.WingBoots), 0
    )


def relic_held(game_state: GameState, relic_name: RelicName) -> bool:
    """Return whether the Character holds the Relic `relic_name`.

    Parameters
    ----------
    game_state : GameState
        The state to read.
    relic_name : RelicName
        The Relic to look for.

    Returns
    -------
    bool
        Whether it is held.
    """
    return any(relic.name == relic_name for relic in game_state.relics)


def _next_row_candidate_cols(
    rooms: list[list[Room | None]], y_current: int | None, x_current: int | None, winged: bool
) -> list[int]:
    """Return the columns of reachable Rooms in the next Map row.

    Parameters
    ----------
    rooms : list[list[Room | None]]
        The Map's rows and columns, with None for absent Rooms.
    y_current : int | None
        The current row, or None before entering the Map.
    x_current : int | None
        The current column, or None before entering the Map.
    winged : bool
        Whether charged Wing Boots allow every Room in the next row.

    Returns
    -------
    list[int]
        Reachable columns, empty beyond the last Map row.
    """
    y_next = 0 if y_current is None else y_current + 1
    if y_next >= MAP_HEIGHT:
        return []

    if y_current is None or winged:
        return [x for x in range(MAP_WIDTH) if rooms[y_next][x] is not None]

    # Agent always stands on a real Room; cur is never None here, so fail loud if it is
    room_curr = rooms[y_current][x_current]
    if room_curr is None:
        raise ValueError(f"Current room ({y_current}, {x_current}) is None")

    for x_next in room_curr.edges:
        if not (0 <= x_next < MAP_WIDTH and rooms[y_next][x_next] is not None):
            raise ValueError(f"Edge ({y_current}, {x_current})->({y_next}, {x_next}) has no room")

    return list(room_curr.edges)


def get_room_entities(game_state: GameState) -> list[Room]:
    """Return the Rooms a RoomSelect can name, in column order: the Room tokens' rank order.

    Parameters
    ----------
    game_state : GameState
        The state to read.

    Returns
    -------
    list[Room]
        The next Map row's Rooms the Character can move to: all of them before the first Room
        or with charged Wing Boots, else the ones the current Room's edges lead to; empty when
        there is no next row.
    """
    map_ = game_state.map
    rooms = map_.rooms
    y_current = map_.y_current
    winged = _get_wing_boots_charges(game_state) > 0
    cols = _next_row_candidate_cols(rooms, y_current, map_.x_current, winged)
    if not cols:
        return []

    y_next = 0 if y_current is None else y_current + 1
    return [room for x, room in enumerate(rooms[y_next]) if x in cols and room is not None]


def get_room_burning(rooms: list[list[Room | None]]) -> Room | None:
    """Return the burning Room, whose won fight grants the Emerald Key.

    Parameters
    ----------
    rooms : list[list[Room | None]]
        The Map's Rooms, by row and column.

    Returns
    -------
    Room | None
        The burning Room, or None when the Map has none.
    """
    return next((room for row in rooms for room in row if room is not None and room.burning), None)


def get_burning_path(game_state: GameState) -> frozenset[tuple[int, int]]:
    """Return the Rooms that still lead to the burning Room within the Wing Boots charges left.

    Only while the Emerald Key can be taken: with the final act on and the key not held, on the
    Maps of Acts 1 to 3, which hold a burning Room. The key masks force the path in Act 3 alone. A
    Room counts when the cheapest way to it from the current position plus the cheapest way from it
    to the burning Room burn no more charges than are left. Each move onto a Room not in the
    current Room's edges burns one; the act's first move is free. Cached by Map, current Room and
    charges, all it depends on.

    Parameters
    ----------
    game_state : GameState
        The state to read.

    Returns
    -------
    frozenset[tuple[int, int]]
        The Rooms' (y, x), the current Room's included while it counts; empty when the key can't
        be taken or the burning Room is out of reach.
    """
    if not game_state.final_act or game_state.key_emerald:
        return frozenset()

    map_ = game_state.map
    wb_charges_left = _get_wing_boots_charges(game_state)
    cache_key = (map_.identity_hash, map_.y_current, map_.x_current, wb_charges_left)
    burning_path = _BURNING_PATH_CACHE.get(cache_key)
    if burning_path is None:
        burning_path = _get_burning_path_uncached(map_, wb_charges_left)
        if len(_BURNING_PATH_CACHE) >= _BURNING_PATH_CACHE_MAX:
            _BURNING_PATH_CACHE.clear()

        _BURNING_PATH_CACHE[cache_key] = burning_path

    return burning_path


def _get_burning_path_uncached(map_: Map, wb_charges_left: int) -> frozenset[tuple[int, int]]:
    """Compute `get_burning_path` on `map_` from its current Room.

    Parameters
    ----------
    map_ : Map
        The act's Map.
    wb_charges_left : int
        The Wing Boots charges left.

    Returns
    -------
    frozenset[tuple[int, int]]
        The Rooms' (y, x); empty when the Map has no burning Room or it is out of reach.
    """
    rooms = map_.rooms
    room_burning = get_room_burning(rooms)
    if room_burning is None:
        return frozenset()

    # Forward from the current Room, or from every first-row Room at the act's start
    y_current, x_current = map_.y_current, map_.x_current
    wb_charges_from = [[math.inf] * MAP_WIDTH for _ in rooms]
    if y_current is None or x_current is None:
        y_first = 0
        for room in rooms[0]:
            if room is not None:
                wb_charges_from[0][room.x] = 0
    
    elif y_current <= room_burning.y:
        y_first = y_current
        wb_charges_from[y_current][x_current] = 0
    
    else:
        return frozenset()

    for y in range(y_first + 1, room_burning.y + 1):
        for room in rooms[y - 1]:
            if room is None or wb_charges_from[y - 1][room.x] == math.inf:
                continue

            edges = room.edges  # Bound once: each read clones the list
            for room_next in rooms[y]:
                if room_next is not None:
                    wb_charges_from[y][room_next.x] = min(
                        wb_charges_from[y][room_next.x],
                        wb_charges_from[y - 1][room.x] + (room_next.x not in edges),
                    )

    wb_charges_to = _wing_boots_charges_to_room_burning(rooms, room_burning)
    return frozenset(
        (y, room.x)
        for y in range(y_first, room_burning.y + 1)
        for room in rooms[y]
        if room is not None
        and wb_charges_from[y][room.x] + wb_charges_to[y][room.x] <= wb_charges_left
    )


def _wing_boots_charges_to_room_burning(
    rooms: list[list[Room | None]], room_burning: Room
) -> list[list[float]]:
    """Return the Wing Boots charges each Room needs to reach the burning Room.

    Each move onto a Room not in the current Room's edges burns one.

    Parameters
    ----------
    rooms : list[list[Room | None]]
        The Map's Rooms, by row and column.
    room_burning : Room
        The burning Room.

    Returns
    -------
    list[list[float]]
        The charges, by row and column; inf where the burning Room can't be reached, as from any
        other Room on its row or past it.
    """
    # Initialize grid
    wb_charges = [[math.inf] * MAP_WIDTH for _ in rooms]
    wb_charges[room_burning.y][room_burning.x] = 0

    # Back from the burning Room's row: each Room takes its cheapest move into the next row
    for y in range(room_burning.y - 1, -1, -1):
        for room in rooms[y]:
            if room is not None:
                edges = room.edges  # Bound once: each read clones the list
                wb_charges[y][room.x] = min(
                    charges_next + (x_next not in edges)
                    for x_next, charges_next in enumerate(wb_charges[y + 1])
                )
                
    return wb_charges


# A token kind's getter, with the reader of the context the getter takes
type _GetterInContext = tuple[
    Callable[[GameState], Combat | Event | Reward | Shop | GameState | None],
    Callable[[Any], Sequence[Entity]],
]

# Each token kind's getter, with the reader of the context it takes
_GETTER_OF_TOKEN_KIND: dict[TokenKind, _GetterInContext] = {
    TokenKind.CARD_MATCH_FLIPPED: (_read_event, _get_card_match_flipped),
    TokenKind.CARD_MATCH_PAIRS: (_read_event, _get_card_match_pairs),
    TokenKind.CARD_PILE_DECK: (_read_game_state, _get_card_pile_deck),
    TokenKind.CARD_PILE_DISCARD: (_read_combat, _get_card_pile_discard),
    TokenKind.CARD_PILE_DISCOVER: (_read_combat, _get_card_pile_discover),
    TokenKind.CARD_PILE_DRAW: (_read_combat, _get_card_pile_draw),
    TokenKind.CARD_PILE_EVENT_ROLL: (_read_event, _get_card_pile_event_roll),
    TokenKind.CARD_PILE_EXHAUST: (_read_combat, _get_card_pile_exhaust),
    TokenKind.CARD_PILE_HAND: (_read_combat, _get_card_pile_hand),
    TokenKind.CARD_PILE_NIGHTMARE: (_read_combat, _get_card_pile_nightmare),
    TokenKind.CARD_PILE_SELECTED: (_read_game_state, _get_card_pile_selected),
    TokenKind.CARD_PILE_STASIS: (_read_combat, _get_card_pile_stasis),
    TokenKind.CARD_PLAY_QUEUE: (_read_combat, _get_card_play_queue),
    TokenKind.CARD_REWARD_0: (_read_reward, _get_card_reward_0),
    TokenKind.CARD_REWARD_1: (_read_reward, _get_card_reward_1),
    TokenKind.CARD_REWARD_2: (_read_reward, _get_card_reward_2),
    TokenKind.CARD_REWARD_3: (_read_reward, _get_card_reward_3),
    TokenKind.CARD_REWARD_4: (_read_reward, _get_card_reward_4),
    TokenKind.CARD_SHOP: (_read_shop, _get_card_shop),
    TokenKind.CHARACTER: (_read_game_state, _get_character),
    TokenKind.EVENT_OPTIONS: (_read_event, _get_event_options),
    TokenKind.MONSTERS: (_read_combat, _get_monsters),
    TokenKind.POTION_EVENT_ROLL: (_read_event, _get_potion_event_roll),
    TokenKind.POTION_OWNED: (_read_game_state, _get_potion_owned),
    TokenKind.POTION_REWARD: (_read_reward, _get_potion_reward),
    TokenKind.POTION_SHOP: (_read_shop, _get_potion_shop),
    TokenKind.RELIC_EVENT_ROLL: (_read_event, _get_relic_event_roll),
    TokenKind.RELIC_OWNED: (_read_game_state, _get_relic_owned),
    TokenKind.RELIC_REWARD: (_read_reward, _get_relic_reward),
    TokenKind.RELIC_SHOP: (_read_shop, _get_relic_shop),
    TokenKind.ROOMS: (_read_game_state, get_room_entities),
}

# Every reader a getter is paired with
_READ_CONTEXTS = frozenset(read_context for read_context, _ in _GETTER_OF_TOKEN_KIND.values())

# Each token kind in window order, with its context's reader and its getter
_GETTERS_IN_WINDOW_ORDER = tuple(
    (token_kind, *_GETTER_OF_TOKEN_KIND[token_kind]) for token_kind in TOKEN_KINDS_IN_WINDOW_ORDER
)


def _check_tables() -> None:
    """Check that each token kind's getter takes the context its reader returns."""
    for token_kind, (read_context, get_entities) in _GETTER_OF_TOKEN_KIND.items():
        type_returned = signature(read_context).return_annotation
        (parameter,) = signature(get_entities).parameters.values()
        if parameter.annotation not in (type_returned, *get_args(type_returned)):
            raise RuntimeError(
                f"{token_kind.name}'s getter takes a {parameter.annotation.__name__}, which "
                f"`{read_context.__name__}` does not return"
            )


_check_tables()
