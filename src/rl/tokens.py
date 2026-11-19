"""The engine collections behind tokens, their caps, and a state's tokens in window order."""

import math
import warnings
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Sequence
from typing import cast

from slai import Card
from slai import Character
from slai import EventOption
from slai import GameState
from slai import LocationOverworld
from slai import LocationStart
from slai import Map
from slai import Monster
from slai import Potion
from slai import Relic
from slai import RelicName
from slai import Room

from src.rl.constants import COLLECTION_KINDS_CARD
from src.rl.constants import COLLECTION_KINDS_CARD_REWARDS
from src.rl.constants import COLLECTION_KINDS_IN_WINDOW_ORDER
from src.rl.constants import COLLECTION_KINDS_POTION
from src.rl.constants import COLLECTION_KINDS_RELIC
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
from src.rl.types import CollectionKind
from src.rl.types import Entity
from src.rl.types import Token


__all__ = [
    "get_entities_for_collection_kind",
    "iter_tokens",
    "get_character",
    "get_entities",
    "get_relic_entities",
    "get_potion_entities",
    "get_event_option_entities",
    "get_monster_entities",
    "relic_held",
    "get_y_x_current",
    "get_rooms",
    "get_room_entities",
    "get_room_burning",
    "get_burning_path",
]


_REWARD_BUNDLE_OF_COLLECTION_KIND = {
    collection_kind: idx_bundle
    for idx_bundle, collection_kind in enumerate(COLLECTION_KINDS_CARD_REWARDS)
}

# The collection kinds that have already warned about overflow
_WARNED_OVERFLOW: set[CollectionKind] = set()

# The burning path by Map, current Room and Wing Boots charges left
_BURNING_PATH_CACHE: dict[tuple[int | None, ...], frozenset[tuple[int, int]]] = {}
_BURNING_PATH_CACHE_MAX = 100_000


def get_entities_for_collection_kind(
    collection_kind: CollectionKind, game_state: GameState
) -> Sequence[Entity]:
    """Return the Entities of a collection kind's tokens in rank order.

    Encoding and action decoding use these same lists, so an Entity's id and its token share
    one placement.

    Parameters
    ----------
    collection_kind : CollectionKind
        The collection kind.
    game_state : GameState
        The state to read.

    Returns
    -------
    Sequence[Entity]
        One Entity per token of the collection kind, in rank order.
    """
    if collection_kind in COLLECTION_KINDS_CARD:
        return _get_card_entities(collection_kind, game_state)
    if collection_kind in COLLECTION_KINDS_RELIC:
        return get_relic_entities(collection_kind, game_state)
    if collection_kind in COLLECTION_KINDS_POTION:
        return get_potion_entities(collection_kind, game_state)
    match collection_kind:
        case CollectionKind.MONSTERS:
            return get_monster_entities(game_state)
        case CollectionKind.EVENT_OPTIONS:
            return get_event_option_entities(game_state)
        case CollectionKind.CHARACTER:
            return [get_character(game_state)]
        case CollectionKind.ROOMS:
            return get_room_entities(game_state)
        case _:
            raise ValueError(f"Unsupported `CollectionKind`: {collection_kind}")


def iter_tokens(game_state: GameState) -> Iterator[Token]:
    """Yield a state's tokens in the shared order used by encoding and action decoding.

    One token per Entity id, at the id's first sighting in `COLLECTION_KINDS_IN_WINDOW_ORDER` and
    then collection order. Only ids merge: a Combat's copy of a deck Card is an Entity of its own.

    Parameters
    ----------
    game_state : GameState
        The state whose capped collections supply the tokens.

    Yields
    ------
    Token
        Each Entity's collection kinds, those of the collections holding it in window order, and
        the Entity, in the order of the Entities' first sightings.
    """
    # Each Entity's token by its id, in the order of first sightings
    token_of_entity_id: dict[int, Token] = {}
    for collection_kind in COLLECTION_KINDS_IN_WINDOW_ORDER:
        for entity in get_entities_for_collection_kind(collection_kind, game_state):
            # The first sighting places the token; each collection adds its collection kind once
            entity_id = entity.id
            token = token_of_entity_id.get(entity_id)
            if token is None:
                token_of_entity_id[entity_id] = ((collection_kind,), entity)
            elif collection_kind not in token[0]:
                token_of_entity_id[entity_id] = ((*token[0], collection_kind), token[1])

    yield from token_of_entity_id.values()


def get_character(game_state: GameState) -> Character:
    """Return the Character.

    Parameters
    ----------
    game_state : GameState
        The state to read.

    Returns
    -------
    Character
        The state's Character, from its Entities.
    """
    (character,) = get_entities(game_state, [game_state.id_character], Character)
    return character


def get_entities[TEntity: Entity](
    game_state: GameState, ids: Iterable[int], entity_type: type[TEntity]
) -> list[TEntity]:
    """Return the Entities with ids `ids`, in their order.

    Parameters
    ----------
    game_state : GameState
        The state whose Entities hold them.
    ids : Iterable[int]
        The Entities' ids, as a collection lists them.
    entity_type : type[TEntity]
        The Entity class the ids name, which types the result.

    Returns
    -------
    list[TEntity]
        One Entity per id.
    """
    entities = game_state.entities
    return cast(list[TEntity], [entities[id_entity] for id_entity in ids])


def _capped(collection_kind: CollectionKind, cards: list[Card], cap: int) -> list[Card]:
    """Cut `cards` to `cap`, warning once per collection kind when that drops any.

    Parameters
    ----------
    collection_kind : CollectionKind
        The Cards' collection kind.
    cards : list[Card]
        The Cards to cut.
    cap : int
        The collection kind's cap.

    Returns
    -------
    list[Card]
        The first `cap` Cards.
    """
    if len(cards) > cap and collection_kind not in _WARNED_OVERFLOW:
        _WARNED_OVERFLOW.add(collection_kind)
        warnings.warn(
            f"{collection_kind.name} holds {len(cards)} cards; the ones past its cap of {cap} are "
            "dropped"
        )
    return cards[:cap]


def _get_card_entities(collection_kind: CollectionKind, game_state: GameState) -> list[Card]:
    """Return the GameState Card pile a Card collection kind names, truncated to the pile's cap.

    Used by get_entities_for_collection_kind. A Card past the cap has no token, so masks cannot
    name it.

    Parameters
    ----------
    collection_kind : CollectionKind
        A Card collection kind.
    game_state : GameState
        The state to read.

    Returns
    -------
    list[Card]
        The pile's Cards in engine order, the draw pile's known Cards by depth and then the rest
        by id; empty when the state has no such pile.
    """
    combat = game_state.combat
    match collection_kind:
        case CollectionKind.CARD_PILE_HAND:
            return (
                get_entities(game_state, combat.id_card_pile_hand[:MAX_SIZE_CARD_PILE_HAND], Card)
                if combat is not None
                else []
            )
        case CollectionKind.CARD_PILE_DRAW:
            if combat is None:
                return []

            # Known Cards by depth, then the rest by id: the order hides what the player can't see
            cards_draw = sorted(
                get_entities(game_state, combat.id_card_pile_draw, Card),
                key=lambda card: (card.draw_depth is None, card.draw_depth or 0, card.id),
            )
            return _capped(collection_kind, cards_draw, MAX_SIZE_CARD_PILE_DRAW)
        case CollectionKind.CARD_PILE_DISCARD:
            if combat is None:
                return []
            return _capped(
                collection_kind,
                get_entities(game_state, combat.id_card_pile_discard, Card),
                MAX_SIZE_CARD_PILE_DISCARD,
            )
        case CollectionKind.CARD_PILE_EXHAUST:
            return (
                get_entities(
                    game_state, combat.id_card_pile_exhaust[:MAX_SIZE_CARD_PILE_EXHAUST], Card
                )
                if combat is not None
                else []
            )
        case CollectionKind.CARD_PILE_STASIS:
            if combat is None:
                return []

            # Slot-parallel to `id_monsters`: an empty slot or a Monster holding no Card is None
            id_cards = [
                id_card
                for id_monster, id_card in zip(combat.id_monsters, combat.id_card_pile_stasis)
                if id_monster is not None and id_card is not None
            ]
            return get_entities(game_state, id_cards[:MAX_SIZE_CARD_PILE_STASIS], Card)
        case CollectionKind.CARD_PILE_DECK:
            return _capped(
                collection_kind,
                get_entities(game_state, game_state.id_card_pile_deck, Card),
                MAX_SIZE_CARD_PILE_DECK,
            )
        case CollectionKind.CARD_PILE_DISCOVER:
            return (
                get_entities(
                    game_state, combat.id_card_pile_discover[:MAX_SIZE_CARD_PILE_DISCOVER], Card
                )
                if combat is not None
                else []
            )
        case (
            CollectionKind.CARD_REWARD_0
            | CollectionKind.CARD_REWARD_1
            | CollectionKind.CARD_REWARD_2
            | CollectionKind.CARD_REWARD_3
            | CollectionKind.CARD_REWARD_4
        ):
            reward = game_state.reward
            idx_bundle = _REWARD_BUNDLE_OF_COLLECTION_KIND[collection_kind]
            if reward is None or idx_bundle >= len(reward.id_cards):
                return []

            # Bundle 0 must hold The Library's single wide roll
            cap = MAX_CARD_REWARD_ROLL if idx_bundle == 0 else MAX_COMBAT_CARD_REWARD
            return get_entities(game_state, reward.id_cards[idx_bundle][:cap], Card)
        case CollectionKind.CARD_SHOP:
            shop = game_state.shop
            return (
                get_entities(game_state, shop.id_cards[:MAX_SIZE_SHOP_CARDS], Card)
                if shop is not None
                else []
            )
        case CollectionKind.CARD_PILE_EVENT_ROLL:
            event = game_state.event
            return (
                get_entities(
                    game_state, event.id_card_pile_event_roll[:MAX_SIZE_CARD_PILE_EVENT_ROLL], Card
                )
                if event is not None
                else []
            )
        case CollectionKind.CARD_MATCH_FLIPPED:
            event = game_state.event
            id_card = event.id_card_match_flipped if event is not None else None
            return get_entities(game_state, [id_card], Card) if id_card is not None else []
        case CollectionKind.CARD_MATCH_PAIRS:
            event = game_state.event
            return (
                get_entities(
                    game_state, event.id_card_match_pairs[:MAX_SIZE_CARD_MATCH_PAIRS], Card
                )
                if event is not None
                else []
            )
        case CollectionKind.CARD_PLAY_QUEUE:
            id_cards = [card_play.id_card for card_play in game_state.card_play_queue]
            return _capped(
                collection_kind, get_entities(game_state, id_cards, Card), MAX_SIZE_CARD_PLAY_QUEUE
            )
        case CollectionKind.CARD_PILE_NIGHTMARE:
            if combat is None:
                return []
            id_cards = [id_card for id_card, _ in combat.id_card_pile_nightmare]
            return _capped(
                collection_kind,
                get_entities(game_state, id_cards, Card),
                MAX_SIZE_CARD_PILE_NIGHTMARE,
            )
        case CollectionKind.CARD_PILE_SELECTED:
            return get_entities(
                game_state, game_state.id_selected[:MAX_SIZE_CARD_PILE_SELECTED], Card
            )
        case _:
            raise ValueError(f"Unsupported `CollectionKind`: {collection_kind}")


def get_relic_entities(collection_kind: CollectionKind, game_state: GameState) -> list[Relic]:
    """Return the capped Relic collection named by a collection kind.

    Parameters
    ----------
    collection_kind : CollectionKind
        A Relic collection kind.
    game_state : GameState
        The state to read.

    Returns
    -------
    list[Relic]
        The collection's Relics in engine order, empty when the collection is absent.
    """
    # Each collection truncated to its own cap: a Relic past it has no token
    match collection_kind:
        case CollectionKind.RELIC_OWNED:
            return get_entities(game_state, game_state.id_relics[:MAX_RELICS], Relic)
        case CollectionKind.RELIC_REWARD:
            reward = game_state.reward
            return (
                get_entities(game_state, reward.id_relics[:MAX_RELIC_REWARDS], Relic)
                if reward is not None
                else []
            )
        case CollectionKind.RELIC_SHOP:
            shop = game_state.shop
            return (
                get_entities(game_state, shop.id_relics[:MAX_SHOP_RELICS], Relic)
                if shop is not None
                else []
            )
        case CollectionKind.RELIC_EVENT_ROLL:
            event = game_state.event
            return (
                get_entities(
                    game_state, event.id_relic_event_roll[:MAX_SIZE_RELIC_EVENT_ROLL], Relic
                )
                if event is not None
                else []
            )
        case _:
            raise ValueError(f"Not a Relic collection kind: {collection_kind}")


def get_potion_entities(collection_kind: CollectionKind, game_state: GameState) -> list[Potion]:
    """Return the capped Potion collection named by a collection kind.

    Parameters
    ----------
    collection_kind : CollectionKind
        A Potion collection kind.
    game_state : GameState
        The state to read.

    Returns
    -------
    list[Potion]
        The collection's Potions in engine order, empty when the collection is absent.
    """
    # Each collection truncated to its own cap (all engine-bounded, so never lossy)
    match collection_kind:
        case CollectionKind.POTION_OWNED:
            return get_entities(game_state, game_state.id_potions[:MAX_POTION_SLOTS], Potion)
        case CollectionKind.POTION_REWARD:
            reward = game_state.reward
            return (
                get_entities(game_state, reward.id_potions[:MAX_POTION_REWARDS], Potion)
                if reward is not None
                else []
            )
        case CollectionKind.POTION_SHOP:
            shop = game_state.shop
            return (
                get_entities(game_state, shop.id_potions[:MAX_SHOP_POTIONS], Potion)
                if shop is not None
                else []
            )
        case CollectionKind.POTION_EVENT_ROLL:
            event = game_state.event
            return (
                get_entities(
                    game_state, event.id_potion_event_roll[:MAX_SIZE_POTION_EVENT_ROLL], Potion
                )
                if event is not None
                else []
            )
        case _:
            raise ValueError(f"Not a Potion collection kind: {collection_kind}")


def get_event_option_entities(game_state: GameState) -> list[EventOption]:
    """Return the options behind the option tokens, in slot order, cut to MAX_EVENT_OPTIONS.

    Parameters
    ----------
    game_state : GameState
        The state to read.

    Returns
    -------
    list[EventOption]
        The Event's first MAX_EVENT_OPTIONS options; empty outside Events.
    """
    event = game_state.event
    return (
        get_entities(game_state, event.id_event_options[:MAX_EVENT_OPTIONS], EventOption)
        if event is not None
        else []
    )


def get_monster_entities(game_state: GameState) -> list[Monster]:
    """Return the Monsters behind the Monster tokens, in slot order.

    Parameters
    ----------
    game_state : GameState
        The state to read.

    Returns
    -------
    list[Monster]
        The Combat's Monsters; empty outside Combat.
    """
    combat = game_state.combat
    if combat is None:
        return []

    # The roster's slots hold None where empty
    id_monsters = [id_monster for id_monster in combat.id_monsters if id_monster is not None]
    return get_entities(game_state, id_monsters, Monster)


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
        (
            relic.counter
            for relic in get_entities(game_state, game_state.id_relics, Relic)
            if relic.name == RelicName.WingBoots
        ),
        0,
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
    return any(
        relic.name == relic_name for relic in get_entities(game_state, game_state.id_relics, Relic)
    )


def get_y_x_current(map_: Map) -> tuple[int | None, int | None]:
    """Return the current Room's row and column, from the Map's location.

    Parameters
    ----------
    map_ : Map
        The act's Map.

    Returns
    -------
    y_current : int | None
        The current row: None at the act's start, MAP_HEIGHT from the boss's Room on.
    x_current : int | None
        The current column: None at the act's start, 0 from the boss's Room on.
    """
    location = map_.location
    if isinstance(location, LocationOverworld):
        return location.y, location.x

    if isinstance(location, LocationStart):
        return None, None

    # The boss's Room, its treasure and the victory sit past the last row
    return MAP_HEIGHT, 0


def get_rooms(game_state: GameState) -> list[list[Room | None]]:
    """Return the Map's Rooms by row and column.

    Parameters
    ----------
    game_state : GameState
        The state whose Entities hold the Rooms.

    Returns
    -------
    list[list[Room | None]]
        Each cell's Room, None where the Map has none.
    """
    # The grid holds None where the Map has no Room, which get_entities can't type
    entities = game_state.entities
    return cast(
        list[list[Room | None]],
        [
            [None if id_room is None else entities[id_room] for id_room in row]
            for row in game_state.map.id_rooms
        ],
    )


def _next_row_candidate_cols(
    game_state: GameState, y_current: int | None, x_current: int | None, winged: bool
) -> list[int]:
    """Return the columns of reachable Rooms in the next Map row.

    Parameters
    ----------
    game_state : GameState
        The state whose Map to read.
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
    id_rooms = game_state.map.id_rooms
    y_next = 0 if y_current is None else y_current + 1
    if y_next >= MAP_HEIGHT:
        return []

    if y_current is None or winged:
        return [x for x in range(MAP_WIDTH) if id_rooms[y_next][x] is not None]

    # Agent always stands on a real Room; cur is never None here, so fail loud if it is
    id_room_curr = id_rooms[y_current][x_current]
    if id_room_curr is None:
        raise ValueError(f"Current room ({y_current}, {x_current}) is None")

    (room_curr,) = get_entities(game_state, [id_room_curr], Room)
    for x_next in room_curr.edges:
        if not (0 <= x_next < MAP_WIDTH and id_rooms[y_next][x_next] is not None):
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
    y_current, x_current = get_y_x_current(map_)
    winged = _get_wing_boots_charges(game_state) > 0
    cols = _next_row_candidate_cols(game_state, y_current, x_current, winged)
    if not cols:
        return []

    y_next = 0 if y_current is None else y_current + 1
    id_rooms_next = [
        id_room
        for x, id_room in enumerate(map_.id_rooms[y_next])
        if x in cols and id_room is not None
    ]
    return get_entities(game_state, id_rooms_next, Room)


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
    y_current, x_current = get_y_x_current(map_)
    wb_charges_left = _get_wing_boots_charges(game_state)
    cache_key = (map_.identity_hash, y_current, x_current, wb_charges_left)
    burning_path = _BURNING_PATH_CACHE.get(cache_key)
    if burning_path is None:
        burning_path = _get_burning_path_uncached(game_state, wb_charges_left)
        if len(_BURNING_PATH_CACHE) >= _BURNING_PATH_CACHE_MAX:
            _BURNING_PATH_CACHE.clear()

        _BURNING_PATH_CACHE[cache_key] = burning_path

    return burning_path


def _get_burning_path_uncached(
    game_state: GameState, wb_charges_left: int
) -> frozenset[tuple[int, int]]:
    """Compute `get_burning_path` on the state's Map from its current Room.

    Parameters
    ----------
    game_state : GameState
        The state whose Map to read.
    wb_charges_left : int
        The Wing Boots charges left.

    Returns
    -------
    frozenset[tuple[int, int]]
        The Rooms' (y, x); empty when the Map has no burning Room or it is out of reach.
    """
    rooms = get_rooms(game_state)
    room_burning = get_room_burning(rooms)
    if room_burning is None:
        return frozenset()

    # Forward from the current Room, or from every first-row Room at the act's start
    y_current, x_current = get_y_x_current(game_state.map)
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
