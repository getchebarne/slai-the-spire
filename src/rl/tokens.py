"""The engine collections behind tokens, their caps, and a state's tokens in window order."""

import math
import warnings
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Sequence
from inspect import formatannotation
from inspect import signature
from typing import Any, cast, get_args

from slai import Card
from slai import Character
from slai import Combat
from slai import Event
from slai import GameState
from slai import LocationOverworld
from slai import LocationStart
from slai import Map
from slai import Monster
from slai import Relic
from slai import RelicName
from slai import Reward
from slai import Room
from slai import Shop

from src.rl.constants import COLLECTION_KINDS_IN_WINDOW_ORDER
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
    "iter_tokens",
    "get_character",
    "get_entities",
    "get_monster_entities",
    "relic_held",
    "get_y_x_current",
    "get_rooms",
    "get_room_entities",
    "get_room_burning",
    "get_burning_path",
]


# The collection kinds that have already warned about overflow
_WARNED_OVERFLOW: set[CollectionKind] = set()

# The burning path by Map, current Room and Wing Boots charges left
_BURNING_PATH_CACHE: dict[tuple[int | None, ...], frozenset[tuple[int, int]]] = {}
_BURNING_PATH_CACHE_MAX = 100_000


def iter_tokens(game_state: GameState) -> Iterator[Token]:
    """Yield a state's tokens in the shared order used by encoding and Action decoding.

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
    entities = game_state.entities
    
    # Each Entity's token by its id, in the order of first sightings
    token_of_entity_id: dict[int, Token] = {}
    context_of_reader = {reader: reader(game_state) for reader in _READERS}
    for collection_kind, reader, getter in _GETTERS_IN_WINDOW_ORDER:
        context = context_of_reader[reader]
        if context is not None:
            for entity_id in getter(context):
                # The first sighting places the token
                token = token_of_entity_id.get(entity_id)
                if token is None:
                    token_of_entity_id[entity_id] = ((collection_kind,), entities[entity_id])

                # Append `CollectionKind` sighting
                elif collection_kind not in token[0]:
                    token_of_entity_id[entity_id] = ((*token[0], collection_kind), token[1])

    yield from token_of_entity_id.values()


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
    combat = _read_combat(game_state)
    if combat is None:
        return []

    return get_entities(game_state, _get_monsters(combat), Monster)


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

    return cast(list[TEntity], [entities[entity_id] for entity_id in ids])


def _capped(collection_kind: CollectionKind, ids: list[int], cap: int) -> list[int]:
    """Cut `ids` to `cap`, warning once per collection kind when that drops any.

    Parameters
    ----------
    collection_kind : CollectionKind
        The collection kind of the Entities the ids name.
    ids : list[int]
        The ids to cut.
    cap : int
        The collection kind's cap.

    Returns
    -------
    list[int]
        The first `cap` ids.
    """
    if len(ids) > cap and collection_kind not in _WARNED_OVERFLOW:
        _WARNED_OVERFLOW.add(collection_kind)
        warnings.warn(
            f"{collection_kind.name} holds {len(ids)} Entities; the ones past its cap of {cap} "
            "are dropped"
        )

    return ids[:cap]


def _read_combat(game_state: GameState) -> Combat | None:
    return game_state.combat


def _read_event(game_state: GameState) -> Event | None:
    return game_state.event


def _read_reward(game_state: GameState) -> Reward | None:
    return game_state.reward


def _read_shop(game_state: GameState) -> Shop | None:
    return game_state.shop


# The collection kinds of no context read the state itself
def _read_game_state(game_state: GameState) -> GameState:
    return game_state


def _get_card_match_flipped(event: Event) -> list[int]:
    id_card = event.id_card_match_flipped

    return [id_card] if id_card is not None else []


def _get_card_match_pairs(event: Event) -> list[int]:
    return event.id_card_match_pairs[:MAX_SIZE_CARD_MATCH_PAIRS]


def _get_card_pile_deck(game_state: GameState) -> list[int]:
    return _capped(
        CollectionKind.CARD_PILE_DECK, game_state.id_card_pile_deck, MAX_SIZE_CARD_PILE_DECK
    )


def _get_card_pile_discard(combat: Combat) -> list[int]:
    return _capped(
        CollectionKind.CARD_PILE_DISCARD, combat.id_card_pile_discard, MAX_SIZE_CARD_PILE_DISCARD
    )


def _get_card_pile_discover(combat: Combat) -> list[int]:
    return combat.id_card_pile_discover[:MAX_SIZE_CARD_PILE_DISCOVER]


# The state is the draw pile's context: its draw-depth order reads the Cards from its Entities
def _get_card_pile_draw(game_state: GameState) -> list[int]:
    combat = game_state.combat
    if combat is None:
        return []

    # Known Cards by depth, then the rest by id: the order hides what the player can't see
    cards_draw = sorted(
        get_entities(game_state, combat.id_card_pile_draw, Card),
        key=lambda card: (card.draw_depth is None, card.draw_depth or 0, card.id),
    )

    return _capped(
        CollectionKind.CARD_PILE_DRAW, [card.id for card in cards_draw], MAX_SIZE_CARD_PILE_DRAW
    )


def _get_card_pile_event_roll(event: Event) -> list[int]:
    return event.id_card_pile_event_roll[:MAX_SIZE_CARD_PILE_EVENT_ROLL]


def _get_card_pile_exhaust(combat: Combat) -> list[int]:
    return _capped(
        CollectionKind.CARD_PILE_EXHAUST, combat.id_card_pile_exhaust, MAX_SIZE_CARD_PILE_EXHAUST
    )


def _get_card_pile_hand(combat: Combat) -> list[int]:
    return combat.id_card_pile_hand[:MAX_SIZE_CARD_PILE_HAND]


def _get_card_pile_nightmare(combat: Combat) -> list[int]:
    return _capped(
        CollectionKind.CARD_PILE_NIGHTMARE,
        [id_card for id_card, _ in combat.id_card_pile_nightmare],
        MAX_SIZE_CARD_PILE_NIGHTMARE,
    )


def _get_card_pile_selected(game_state: GameState) -> list[int]:
    return game_state.id_selected[:MAX_SIZE_CARD_PILE_SELECTED]


def _get_card_pile_stasis(combat: Combat) -> list[int]:
    # Slot-parallel to `id_monsters`: an empty slot or a Monster holding no Card is None
    id_cards = [
        id_card
        for id_monster, id_card in zip(combat.id_monsters, combat.id_card_pile_stasis)
        if id_monster is not None and id_card is not None
    ]

    return id_cards[:MAX_SIZE_CARD_PILE_STASIS]


def _get_card_play_queue(game_state: GameState) -> list[int]:
    return _capped(
        CollectionKind.CARD_PLAY_QUEUE,
        [card_play.id_card for card_play in game_state.card_play_queue],
        MAX_SIZE_CARD_PLAY_QUEUE,
    )


def _get_card_reward_bundle(reward: Reward, idx_bundle: int, cap: int) -> list[int]:
    id_cards = reward.id_cards

    return id_cards[idx_bundle][:cap] if idx_bundle < len(id_cards) else []


def _get_card_reward_0(reward: Reward) -> list[int]:
    # Bundle 0 must hold The Library's single wide roll
    return _get_card_reward_bundle(reward, 0, MAX_CARD_REWARD_ROLL)


def _get_card_reward_1(reward: Reward) -> list[int]:
    return _get_card_reward_bundle(reward, 1, MAX_COMBAT_CARD_REWARD)


def _get_card_reward_2(reward: Reward) -> list[int]:
    return _get_card_reward_bundle(reward, 2, MAX_COMBAT_CARD_REWARD)


def _get_card_reward_3(reward: Reward) -> list[int]:
    return _get_card_reward_bundle(reward, 3, MAX_COMBAT_CARD_REWARD)


def _get_card_reward_4(reward: Reward) -> list[int]:
    return _get_card_reward_bundle(reward, 4, MAX_COMBAT_CARD_REWARD)


def _get_card_shop(shop: Shop) -> list[int]:
    return shop.id_cards[:MAX_SIZE_SHOP_CARDS]


def _get_character(game_state: GameState) -> list[int]:
    return [game_state.id_character]


def _get_event_options(event: Event) -> list[int]:
    return event.id_event_options[:MAX_EVENT_OPTIONS]


def _get_monsters(combat: Combat) -> list[int]:
    # The roster's slots hold None where empty
    return [id_monster for id_monster in combat.id_monsters if id_monster is not None]


def _get_potion_event_roll(event: Event) -> list[int]:
    return event.id_potion_event_roll[:MAX_SIZE_POTION_EVENT_ROLL]


def _get_potion_owned(game_state: GameState) -> list[int]:
    return game_state.id_potions[:MAX_POTION_SLOTS]


def _get_potion_reward(reward: Reward) -> list[int]:
    return reward.id_potions[:MAX_POTION_REWARDS]


def _get_potion_shop(shop: Shop) -> list[int]:
    return shop.id_potions[:MAX_SHOP_POTIONS]


def _get_relic_event_roll(event: Event) -> list[int]:
    return event.id_relic_event_roll[:MAX_SIZE_RELIC_EVENT_ROLL]


def _get_relic_owned(game_state: GameState) -> list[int]:
    return _capped(CollectionKind.RELIC_OWNED, game_state.id_relics, MAX_RELICS)


def _get_relic_reward(reward: Reward) -> list[int]:
    return reward.id_relics[:MAX_RELIC_REWARDS]


def _get_relic_shop(shop: Shop) -> list[int]:
    return shop.id_relics[:MAX_SHOP_RELICS]


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

    if y_current is None or x_current is None or winged:
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
    return get_entities(game_state, _get_id_rooms_next(game_state), Room)


def _get_id_rooms_next(game_state: GameState) -> list[int]:
    """Return the ids of the Rooms a RoomSelect can name, `get_room_entities`' in its order.

    Parameters
    ----------
    game_state : GameState
        The state to read.

    Returns
    -------
    list[int]
        The next Map row's Room ids the Character can move to, in column order.
    """
    map_ = game_state.map
    y_current, x_current = get_y_x_current(map_)
    winged = _get_wing_boots_charges(game_state) > 0
    cols = _next_row_candidate_cols(game_state, y_current, x_current, winged)
    if not cols:
        return []

    y_next = 0 if y_current is None else y_current + 1

    return [
        id_room
        for x, id_room in enumerate(map_.id_rooms[y_next])
        if x in cols and id_room is not None
    ]


# The reader of a collection kind's context, then the collection kind's getter of its Entities' ids
type _GetterInContext = tuple[
    Callable[[GameState], Combat | Event | Reward | Shop | GameState | None],
    Callable[[Any], Sequence[int]],
]

# The reader of each collection kind's context, then the collection kind's getter
_GETTER_OF_COLLECTION_KIND: dict[CollectionKind, _GetterInContext] = {
    CollectionKind.CARD_MATCH_FLIPPED: (_read_event, _get_card_match_flipped),
    CollectionKind.CARD_MATCH_PAIRS: (_read_event, _get_card_match_pairs),
    CollectionKind.CARD_PILE_DECK: (_read_game_state, _get_card_pile_deck),
    CollectionKind.CARD_PILE_DISCARD: (_read_combat, _get_card_pile_discard),
    CollectionKind.CARD_PILE_DISCOVER: (_read_combat, _get_card_pile_discover),
    CollectionKind.CARD_PILE_DRAW: (_read_game_state, _get_card_pile_draw),
    CollectionKind.CARD_PILE_EVENT_ROLL: (_read_event, _get_card_pile_event_roll),
    CollectionKind.CARD_PILE_EXHAUST: (_read_combat, _get_card_pile_exhaust),
    CollectionKind.CARD_PILE_HAND: (_read_combat, _get_card_pile_hand),
    CollectionKind.CARD_PILE_NIGHTMARE: (_read_combat, _get_card_pile_nightmare),
    CollectionKind.CARD_PILE_SELECTED: (_read_game_state, _get_card_pile_selected),
    CollectionKind.CARD_PILE_STASIS: (_read_combat, _get_card_pile_stasis),
    CollectionKind.CARD_PLAY_QUEUE: (_read_game_state, _get_card_play_queue),
    CollectionKind.CARD_REWARD_0: (_read_reward, _get_card_reward_0),
    CollectionKind.CARD_REWARD_1: (_read_reward, _get_card_reward_1),
    CollectionKind.CARD_REWARD_2: (_read_reward, _get_card_reward_2),
    CollectionKind.CARD_REWARD_3: (_read_reward, _get_card_reward_3),
    CollectionKind.CARD_REWARD_4: (_read_reward, _get_card_reward_4),
    CollectionKind.CARD_SHOP: (_read_shop, _get_card_shop),
    CollectionKind.CHARACTER: (_read_game_state, _get_character),
    CollectionKind.EVENT_OPTIONS: (_read_event, _get_event_options),
    CollectionKind.MONSTERS: (_read_combat, _get_monsters),
    CollectionKind.POTION_EVENT_ROLL: (_read_event, _get_potion_event_roll),
    CollectionKind.POTION_OWNED: (_read_game_state, _get_potion_owned),
    CollectionKind.POTION_REWARD: (_read_reward, _get_potion_reward),
    CollectionKind.POTION_SHOP: (_read_shop, _get_potion_shop),
    CollectionKind.RELIC_EVENT_ROLL: (_read_event, _get_relic_event_roll),
    CollectionKind.RELIC_OWNED: (_read_game_state, _get_relic_owned),
    CollectionKind.RELIC_REWARD: (_read_reward, _get_relic_reward),
    CollectionKind.RELIC_SHOP: (_read_shop, _get_relic_shop),
    CollectionKind.ROOMS: (_read_game_state, _get_id_rooms_next),
}


def _check_tables() -> None:
    """Check that each collection kind has its own getter, taking its reader's context."""
    if set(_GETTER_OF_COLLECTION_KIND) != set(CollectionKind):
        raise RuntimeError(
            "A CollectionKind has no entry in _GETTER_OF_COLLECTION_KIND: "
            f"{set(CollectionKind) - set(_GETTER_OF_COLLECTION_KIND)}"
        )

    collection_kind_of_getter: dict[Callable[[Any], Sequence[int]], CollectionKind] = {}
    for collection_kind, (reader, getter) in _GETTER_OF_COLLECTION_KIND.items():
        collection_kind_first = collection_kind_of_getter.setdefault(getter, collection_kind)
        if collection_kind_first is not collection_kind:
            raise RuntimeError(
                f"`{getter.__name__}` serves two collection kinds: {collection_kind_first.name} "
                f"and {collection_kind.name}"
            )

        parameters = signature(getter).parameters
        if len(parameters) != 1:
            raise RuntimeError(
                f"{collection_kind.name}'s getter `{getter.__name__}` takes {len(parameters)} "
                "parameters, not its context alone"
            )

        # The reader's return type is its context, or a union of its context and None
        type_returned = signature(reader).return_annotation
        (parameter,) = parameters.values()
        if parameter.annotation not in (type_returned, *get_args(type_returned)):
            raise RuntimeError(
                f"{collection_kind.name}'s getter takes {formatannotation(parameter.annotation)}, "
                f"which `{reader.__name__}` does not return"
            )


_check_tables()

# Every reader a getter is paired with
_READERS = frozenset(reader for reader, _ in _GETTER_OF_COLLECTION_KIND.values())

# Each collection kind in window order, with its context's reader and its getter
_GETTERS_IN_WINDOW_ORDER = tuple(
    (collection_kind, *_GETTER_OF_COLLECTION_KIND[collection_kind])
    for collection_kind in COLLECTION_KINDS_IN_WINDOW_ORDER
)


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

    Only while the Emerald Key can be taken: with the key not held, on the Maps of Acts 1 to 3,
    which hold a burning Room. The key masks force the path in Act 3 alone. A Room counts when the
    cheapest way to it from the current position plus the cheapest way from it to the burning Room
    burn no more charges than are left. Each move onto a Room not in the current Room's edges burns
    one; the act's first move is free. Cached by Map, current Room and charges, all it depends on.

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
    if game_state.key_emerald:
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
