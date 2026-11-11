"""The engine collections behind tokens, their caps and their shared packing order."""

import warnings
from collections.abc import Iterator
from collections.abc import Sequence

from slai import Card
from slai import EventOption
from slai import GameState
from slai import Monster
from slai import Potion
from slai import Relic
from slai import RelicName
from slai import Room

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
from src.rl.constants import TOKEN_KINDS_CARD
from src.rl.constants import TOKEN_KINDS_CARD_REWARDS
from src.rl.constants import TOKEN_KINDS_POTION
from src.rl.constants import TOKEN_KINDS_RELIC
from src.rl.types import Entity
from src.rl.types import Token
from src.rl.types import TokenKind


__all__ = [
    "get_entities_for_token_kind",
    "iter_tokens",
    "get_card_entities",
    "get_relic_entities",
    "get_potion_entities",
    "get_event_option_entities",
    "get_monster_entities",
    "get_room_entities",
]


_REWARD_BUNDLE_OF_TOKEN_KIND = {
    token_kind: bundle for bundle, token_kind in enumerate(TOKEN_KINDS_CARD_REWARDS)
}

# The token kinds that have already warned about overflow
_WARNED_OVERFLOW: set[TokenKind] = set()


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
    if token_kind in TOKEN_KINDS_CARD:
        return get_card_entities(token_kind, game_state)
    if token_kind in TOKEN_KINDS_RELIC:
        return get_relic_entities(token_kind, game_state)
    if token_kind in TOKEN_KINDS_POTION:
        return get_potion_entities(token_kind, game_state)
    match token_kind:
        case TokenKind.MONSTERS:
            return get_monster_entities(game_state)
        case TokenKind.EVENT_OPTIONS:
            return get_event_option_entities(game_state)
        case TokenKind.CHARACTER:
            return [game_state.character]
        case TokenKind.ROOMS:
            return get_room_entities(game_state)
        case _:
            raise ValueError(f"Unsupported `TokenKind`: {token_kind}")


def iter_tokens(game_state: GameState) -> Iterator[Token]:
    """Yield a state's tokens in the shared order used by encoding and action decoding.

    Parameters
    ----------
    game_state : GameState
        The state whose capped collections supply the tokens.

    Yields
    ------
    Token
        Each token kind and Entity, in packing order and then collection order.
    """
    for token_kind in TokenKind:
        for entity in get_entities_for_token_kind(token_kind, game_state):
            yield token_kind, entity


def _capped(
    token_kind: TokenKind, cards: list[Card], cap: int, keep_last: bool = False
) -> list[Card]:
    """Cut `cards` to `cap`, warning once per token kind when that drops any.

    Parameters
    ----------
    token_kind : TokenKind
        The Cards' token kind.
    cards : list[Card]
        The Cards to cut.
    cap : int
        The token kind's cap.
    keep_last : bool, default False
        Keep the last `cap` Cards rather than the first.

    Returns
    -------
    list[Card]
        The first `cap` Cards, or the last `cap` with `keep_last`.
    """
    if len(cards) > cap and token_kind not in _WARNED_OVERFLOW:
        _WARNED_OVERFLOW.add(token_kind)
        warnings.warn(
            f"{token_kind.name} holds {len(cards)} cards; the ones past its cap of {cap} are "
            "dropped"
        )
    return cards[max(0, len(cards) - cap) :] if keep_last else cards[:cap]


def get_card_entities(token_kind: TokenKind, game_state: GameState) -> list[Card]:
    """Return the GameState Card pile a Card token kind names, truncated to the pile's cap.

    Used by get_entities_for_token_kind. A Card past the cap has no token, so masks cannot name
    it.

    Parameters
    ----------
    token_kind : TokenKind
        A Card token kind.
    game_state : GameState
        The state to read.

    Returns
    -------
    list[Card]
        The pile's Cards in engine order, the draw pile's cut to the ones nearest its top; empty
        when the state has no such pile.
    """
    combat = game_state.combat
    match token_kind:
        case TokenKind.CARD_PILE_HAND:
            return combat.card_pile_hand[:MAX_SIZE_CARD_PILE_HAND] if combat is not None else []
        case TokenKind.CARD_PILE_DRAW:
            if combat is None:
                return []

            # The top of the pile, still in engine order (bottom first)
            return _capped(
                token_kind, combat.card_pile_draw, MAX_SIZE_CARD_PILE_DRAW, keep_last=True
            )
        case TokenKind.CARD_PILE_DISCARD:
            if combat is None:
                return []
            return _capped(token_kind, combat.card_pile_discard, MAX_SIZE_CARD_PILE_DISCARD)
        case TokenKind.CARD_PILE_EXHAUST:
            return (
                combat.card_pile_exhaust[:MAX_SIZE_CARD_PILE_EXHAUST] if combat is not None else []
            )
        case TokenKind.CARD_PILE_STASIS:
            if combat is None:
                return []

            # Index-aligned with the Monsters: a Monster holding no Card is None
            cards = [card for card in combat.card_pile_stasis if card is not None]
            return cards[:MAX_SIZE_CARD_PILE_STASIS]
        case TokenKind.CARD_PILE_DECK:
            return _capped(token_kind, game_state.card_pile_deck, MAX_SIZE_CARD_PILE_DECK)
        case TokenKind.CARD_PILE_DISCOVER:
            return (
                combat.card_pile_discover[:MAX_SIZE_CARD_PILE_DISCOVER]
                if combat is not None
                else []
            )
        case (
            TokenKind.CARD_REWARD_0
            | TokenKind.CARD_REWARD_1
            | TokenKind.CARD_REWARD_2
            | TokenKind.CARD_REWARD_3
            | TokenKind.CARD_REWARD_4
        ):
            reward = game_state.reward
            bundle = _REWARD_BUNDLE_OF_TOKEN_KIND[token_kind]
            if reward is None or bundle >= len(reward.cards):
                return []

            # Bundle 0 must hold The Library's single wide roll
            cap = MAX_CARD_REWARD_ROLL if bundle == 0 else MAX_COMBAT_CARD_REWARD
            return reward.cards[bundle][:cap]
        case TokenKind.CARD_SHOP:
            shop = game_state.shop
            return shop.cards[:MAX_SIZE_SHOP_CARDS] if shop is not None else []
        case TokenKind.CARD_PILE_EVENT_ROLL:
            event = game_state.event
            return (
                event.card_pile_event_roll[:MAX_SIZE_CARD_PILE_EVENT_ROLL]
                if event is not None
                else []
            )
        case TokenKind.CARD_MATCH_FLIPPED:
            event = game_state.event
            card = event.card_match_flipped if event is not None else None
            return [card] if card is not None else []
        case TokenKind.CARD_MATCH_PAIRS:
            event = game_state.event
            return event.card_match_pairs[:MAX_SIZE_CARD_MATCH_PAIRS] if event is not None else []
        case TokenKind.CARD_PLAY_QUEUE:
            if combat is None:
                return []
            return _capped(token_kind, combat.card_play_queue, MAX_SIZE_CARD_PLAY_QUEUE)
        case TokenKind.CARD_PILE_NIGHTMARE:
            if combat is None:
                return []
            return _capped(token_kind, combat.card_pile_nightmare, MAX_SIZE_CARD_PILE_NIGHTMARE)
        case TokenKind.CARD_PILE_SELECTED:
            return game_state.card_pile_selected[:MAX_SIZE_CARD_PILE_SELECTED]
        case _:
            raise ValueError(f"Unsupported `TokenKind`: {token_kind}")


def get_relic_entities(token_kind: TokenKind, game_state: GameState) -> list[Relic]:
    """Return the capped Relic collection named by a token kind.

    Parameters
    ----------
    token_kind : TokenKind
        A Relic token kind.
    game_state : GameState
        The state to read.

    Returns
    -------
    list[Relic]
        The collection's Relics in engine order, empty when the collection is absent.
    """
    # Each collection truncated to its own cap: a Relic past it has no token
    match token_kind:
        case TokenKind.RELIC_OWNED:
            return game_state.relics[:MAX_RELICS]
        case TokenKind.RELIC_REWARD:
            reward = game_state.reward
            return reward.relics[:MAX_RELIC_REWARDS] if reward is not None else []
        case TokenKind.RELIC_SHOP:
            shop = game_state.shop
            return shop.relics[:MAX_SHOP_RELICS] if shop is not None else []
        case TokenKind.RELIC_EVENT_ROLL:
            event = game_state.event
            return event.relic_event_roll[:MAX_SIZE_RELIC_EVENT_ROLL] if event is not None else []
        case _:
            raise ValueError(f"Not a Relic token kind: {token_kind}")


def get_potion_entities(token_kind: TokenKind, game_state: GameState) -> list[Potion]:
    """Return the capped Potion collection named by a token kind.

    Parameters
    ----------
    token_kind : TokenKind
        A Potion token kind.
    game_state : GameState
        The state to read.

    Returns
    -------
    list[Potion]
        The collection's Potions in engine order, empty when the collection is absent.
    """
    # Each collection truncated to its own cap (all engine-bounded, so never lossy)
    match token_kind:
        case TokenKind.POTION_OWNED:
            return game_state.potions[:MAX_POTION_SLOTS]
        case TokenKind.POTION_REWARD:
            reward = game_state.reward
            return reward.potions[:MAX_POTION_REWARDS] if reward is not None else []
        case TokenKind.POTION_SHOP:
            shop = game_state.shop
            return shop.potions[:MAX_SHOP_POTIONS] if shop is not None else []
        case TokenKind.POTION_EVENT_ROLL:
            event = game_state.event
            return (
                event.potion_event_roll[:MAX_SIZE_POTION_EVENT_ROLL] if event is not None else []
            )
        case _:
            raise ValueError(f"Not a Potion token kind: {token_kind}")


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
    return event.options[:MAX_EVENT_OPTIONS] if event is not None else []


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
    return combat.monsters if combat is not None else []


def _wing_boots_charged(game_state: GameState) -> bool:
    """Return whether the Character holds Wing Boots with charges left.

    With charges left, the engine offers every next-row Room, not just the edges.

    Parameters
    ----------
    game_state : GameState
        The state to read.

    Returns
    -------
    bool
        True when a Wing Boots Relic's counter is above 0.
    """
    return any(r.name == RelicName.WingBoots and r.counter > 0 for r in game_state.relics)


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
    winged = _wing_boots_charged(game_state)
    cols = _next_row_candidate_cols(rooms, y_current, map_.x_current, winged)
    if not cols:
        return []

    y_next = 0 if y_current is None else y_current + 1
    return [room for x, room in enumerate(rooms[y_next]) if x in cols and room is not None]
