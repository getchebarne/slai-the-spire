import math
import warnings
from typing import cast

import numpy as np
import slai
from slai import ActionType
from slai import Card
from slai import EffectRoomEnter
from slai import EventName
from slai import GameEnv
from slai import PotionName
from slai import RelicName
from slai import Room

from src.rl.constants import MAP_HEIGHT
from src.rl.constants import MAP_WIDTH
from src.rl.encoding.card import card_key
from src.rl.encoding.effect import effect_key
from src.rl.encoding.focus import derive_focus
from src.rl.encoding.focus import draw_order_visible
from src.rl.types import NUM_ACTION_TYPES
from src.rl.types import TOKEN_KINDS_CARD
from src.rl.types import Entity
from src.rl.types import TokenKind


def _card_identity_ids(cards: list[Card]) -> list[int]:
    """Return contiguous group ids over a pile; same-identity Cards share an id (selection dedup).

    Groups by `card_key` and the `effect_key` of the Card's play, discard and draw Effects — the
    Card row plus the Effect rows, the Card's full model-visible identity. Two same-name Cards can
    carry different Effects (Glass Knife decay, Ritual Dagger bumps) and are NOT interchangeable
    selections. Not keyed on the Card object: the engine's __hash__ covers the arena `id`, which
    would make every Card distinct and the dedup a no-op.

    Parameters
    ----------
    cards : list[Card]
        The pile's Cards.

    Returns
    -------
    list[int]
        One group id per Card, in pile order, numbered from 0 in first-seen order.
    """
    seen: dict[tuple, int] = {}
    return [
        seen.setdefault(
            (
                card_key(card),
                effect_key(card.effects_play),
                effect_key(card.effects_discard),
                effect_key(card.effects_draw),
            ),
            len(seen),
        )
        for card in cards
    ]


def _pile_ids(
    game_state: slai.GameState, token_kind: TokenKind, entities: list[Entity]
) -> list[int]:
    """Return the identity group-ids for a token kind's dedup.

    Parameters
    ----------
    game_state : slai.GameState
        The tokens' state; Frozen Eye in it turns the draw pile's dedup off.
    token_kind : TokenKind
        The token kind.
    entities : list[Entity]
        Its tokens' Entities, in rank order.

    Returns
    -------
    list[int]
        One group id per Entity; [] if the token kind doesn't dedup.
    """
    if token_kind not in TOKEN_KINDS_CARD:
        return []

    # Frozen Eye shows the draw order: same-identity copies at different depths are different selections
    if token_kind is TokenKind.CARD_PILE_DRAW and draw_order_visible(game_state):
        return []

    # A Card `TokenKind`'s Entities are Cards
    return _card_identity_ids(cast(list[Card], entities))


def _wasted_potion_discard(game_state: slai.GameState, action: slai.Action) -> bool:
    """Return whether `action` discards a Potion while nothing needs the room.

    No Potion can arrive under Sozu, and with a free belt slot the discard only pays as a reroll
    before drinking an Entropic Brew held in another slot.

    Parameters
    ----------
    game_state : slai.GameState
        The state `action` is legal in.
    action : slai.Action
        One legal action.

    Returns
    -------
    bool
        True for a Potion discard under Sozu, or with a free belt slot and no Entropic Brew in
        another slot.
    """
    if action.action_type is not ActionType.PotionDiscard:
        return False
    if any(relic.name == RelicName.Sozu for relic in game_state.relics):
        return True
    potions = game_state.potions
    return len(potions) < game_state.potion_slots_max and not any(
        potion.name == PotionName.EntropicBrew for potion in potions if potion.id != action.ids[0]
    )


def _get_room_burning(rooms: list[list[Room | None]]) -> Room:
    return next(room for row in rooms for room in row if room is not None and room.burning)


def _wing_boots_charges_to_room_burning(rooms: list[list[Room | None]]) -> list[list[float]]:
    """Return the Wing Boots charges each Room needs to reach the burning Room.

    Each move onto a Room not in the current Room's edges burns one.

    Parameters
    ----------
    rooms : list[list[Room | None]]
        The Map's Rooms, by row and column.

    Returns
    -------
    list[list[float]]
        The charges, by row and column; inf where the burning Room can't be reached, as from any
        other Room on its row or past it.
    """
    room_burning = _get_room_burning(rooms)

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


def _misses_room_burning(game_state: slai.GameState, action: slai.Action) -> bool:
    """Return whether a RoomSelect puts the burning Room out of reach.

    Parameters
    ----------
    game_state : slai.GameState
        The state `action` is legal in.
    action : slai.Action
        A legal RoomSelect.

    Returns
    -------
    bool
        True when the burning Room takes more Wing Boots charges than are left from the selected
        Room, counting the move itself, and doesn't from another Room in its row.
    """
    # Handy references
    map_ = game_state.map
    rooms = map_.rooms
    y_current, x_current = map_.y_current, map_.x_current
    id_room = action.ids[0]

    # Get current and selected Rooms
    room_current = None if y_current is None or x_current is None else rooms[y_current][x_current]
    room_selected = next(
        room for row in rooms for room in row if room is not None and room.id == id_room
    )

    # Number of remaining Wing Boots charges
    wb_charges_left = next(
        (relic.counter for relic in game_state.relics if relic.name == RelicName.WingBoots), 0
    )

    # The move itself: free from the Act's start, else off the current Room's edges it burns one
    edges = range(MAP_WIDTH) if room_current is None else room_current.edges
    wb_charges_row = [
        charges + (x not in edges)
        for x, charges in enumerate(_wing_boots_charges_to_room_burning(rooms)[room_selected.y])
    ]

    return min(wb_charges_row) <= wb_charges_left < wb_charges_row[room_selected.x]


def _escapes_room_burning(game_state: slai.GameState, action: slai.Action) -> bool:
    """Return whether a PotionUse flees the burning Room's fight, which gives no key.

    Parameters
    ----------
    game_state : slai.GameState
        The state `action` is legal in.
    action : slai.Action
        A legal PotionUse.

    Returns
    -------
    bool
        True for a Smoke Bomb in the burning Room.
    """
    id_potion = action.ids[0]
    potion = next(potion for potion in game_state.potions if potion.id == id_potion)
    if potion.name != PotionName.SmokeBomb:
        return False

    map_ = game_state.map
    room_burning = _get_room_burning(map_.rooms)
    return map_.y_current == room_burning.y and map_.x_current == room_burning.x


def _leaves_key_sapphire(game_state: slai.GameState, action: slai.Action) -> bool:
    """Return whether a RoomExit or RewardTakeRelic gives up the Sapphire Key from row 8 on.

    Parameters
    ----------
    game_state : slai.GameState
        The state `action` is legal in.
    action : slai.Action
        A legal RoomExit or RewardTakeRelic.

    Returns
    -------
    bool
        True for a RoomExit at a Chest or while `relic_linked` is set, and for taking the linked
        Relic.
    """
    # Row 8 is always a Chest
    y_current = game_state.map.y_current
    if y_current is None or y_current < 8:
        return False

    # An opened Chest closes for its Reward
    reward = game_state.reward
    if reward is None:
        return game_state.chest is not None

    # Took the linked Relic instead of the key
    relic_linked = reward.relic_linked
    return relic_linked is not None and (
        action.action_type is ActionType.RoomExit or action.ids[0] == relic_linked.id
    )


def _enters_secret_portal(game_state: slai.GameState, action: slai.Action) -> bool:
    """Return whether an EventOptionSelect takes the Secret Portal, which skips the keys' rows.

    Parameters
    ----------
    game_state : slai.GameState
        The state `action` is legal in.
    action : slai.Action
        A legal EventOptionSelect.

    Returns
    -------
    bool
        True when the Event is the Secret Portal and the option's Effects enter a Room.
    """
    event = game_state.event
    if event is None or event.name != EventName.SecretPortal:
        return False

    id_option = action.ids[0]
    option = next(option for option in event.options if option.id == id_option)
    return any(isinstance(effect, EffectRoomEnter) for effect in option.effects)


def _forfeits_key(game_state: slai.GameState, action: slai.Action) -> bool:
    """Return whether `action` gives up a key.

    Only Act 3 can lose a key, with the final act on.

    Parameters
    ----------
    game_state : slai.GameState
        The state `action` is legal in.
    action : slai.Action
        One legal action.

    Returns
    -------
    bool
        True with a key missing in Act 3, for a rest option other than Recall at the last Rest Site
        (Ruby Key), a move that gives up the Sapphire Key from row 8 on, a Room or a Smoke Bomb
        that loses the burning Room (Emerald Key), and entering the Secret Portal (any key).
    """
    if (
        game_state.act != GameEnv.ACT_FINAL - 1
        or not game_state.final_act
        or (game_state.key_emerald and game_state.key_ruby and game_state.key_sapphire)
    ):
        return False

    match action.action_type:
        case (
            ActionType.Rest
            | ActionType.RestDig
            | ActionType.RestLift
            | ActionType.RestSmith
            | ActionType.RestToke
        ):
            # The Map's last row is always a Rest Site
            return not game_state.key_ruby and game_state.map.y_current == MAP_HEIGHT - 1
        case ActionType.RoomExit | ActionType.RewardTakeRelic:
            return not game_state.key_sapphire and _leaves_key_sapphire(game_state, action)
        case ActionType.RoomSelect:
            return not game_state.key_emerald and _misses_room_burning(game_state, action)
        case ActionType.PotionUse:
            return not game_state.key_emerald and _escapes_room_burning(game_state, action)
        case ActionType.EventOptionSelect:
            return _enters_secret_portal(game_state, action)
        case _:
            return False


# Action types already warned about dropped selections — truncation is never silent
_WARNED_DROPPED: set[int] = set()


def encode_batch_masks(
    batch_game_state: list[slai.GameState],
    batch_legal_actions: list[list[slai.Action]],
    batch_tokens: list[list[tuple[TokenKind, Entity]]],
    batch_row_of_id: list[dict[int, int]],
    np_token_kind: np.ndarray,
    np_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode the masks from each state's legal actions.

    Wasted Potion discards and actions that forfeit a key are engine-legal but masked.

    The token pass gives each state's tokens, as (token kind, Entity) in window order, and each
    Entity id's first row. A legal id's bit goes on that row, so the masks read the encoder's
    placement rather than re-deriving it, and an Entity the encoder dropped is unselectable by
    construction.

    Parameters
    ----------
    batch_game_state : list[slai.GameState]
        The states, B of them.
    batch_legal_actions : list[list[slai.Action]]
        Each state's legal actions.
    batch_tokens : list[list[tuple[TokenKind, Entity]]]
        Each state's tokens, as (token kind, Entity) in window order.
    batch_row_of_id : list[dict[int, int]]
        Each state's Entity ids, each mapped to its first row.
    np_token_kind : np.ndarray
        (N,) int16. Each token's token kind.
    np_count : np.ndarray
        (B,) int64. Each state's number of tokens.

    Returns
    -------
    np_legal : np.ndarray
        (N, NUM_ACTION_TYPES) bool. The tokens' legal bits, per action type, deduped by Card
        identity within each token kind; a Monster token's bits mark the action types whose legal
        actions target it.
    np_needs_target : np.ndarray
        (N,) bool. The tokens' needs-target bits: set where a legal action carries a Monster
        target.
    np_legal_action_type : np.ndarray
        (B, NUM_ACTION_TYPES) bool. The states' legal action types.
    """

    # Each state's first row and the end of the last
    np_start = np.concatenate([[0], np.cumsum(np_count)])

    # Initialize the masks: each token's legal action types and its needs-target bit, and each
    # state's legal action types
    np_legal = np.zeros((len(np_token_kind), NUM_ACTION_TYPES), dtype=np.bool_)
    np_needs_target = np.zeros(len(np_token_kind), dtype=np.bool_)
    np_legal_action_type = np.zeros((len(batch_game_state), NUM_ACTION_TYPES), dtype=np.bool_)

    for b, game_state in enumerate(batch_game_state):
        tokens = batch_tokens[b]
        row_of_id = batch_row_of_id[b]
        start, end = int(np_start[b]), int(np_start[b + 1])

        # The (type, token kind) pairs whose legal ids set a bit, for the dedup below
        legal_token_kinds: set[tuple[int, TokenKind]] = set()

        for action in batch_legal_actions[b]:
            if _wasted_potion_discard(game_state, action) or _forfeits_key(game_state, action):
                continue

            action_type = int(action.action_type)
            ids = action.ids  # Bound once: each read clones the list
            if ids:
                row = row_of_id.get(ids[0])
                if row is None:
                    # No token (cap overflow, or the encoder disagrees with the engine):
                    # unselectable, so its type bit stays off
                    if action_type not in _WARNED_DROPPED:
                        _WARNED_DROPPED.add(action_type)
                        warnings.warn(
                            f"Dropped legal action: type={action_type} id {ids[0]} has no token"
                        )
                    continue

                np_legal[row, action_type] = True  # Per ACTION TYPE — legality differs per type
                legal_token_kinds.add((action_type, tokens[row - start][0]))

                # L3: only CardPlay/PotionUse carry a Monster target as ids[1]. The selection's
                # needs-target bit records that it takes one, and the target's token gets the
                # type's bit: a half-dead Monster gets PotionUse's but never CardPlay's
                if len(ids) == 2:
                    np_needs_target[row] = True
                    np_legal[row_of_id[ids[1]], action_type] = True
            np_legal_action_type[
                b, action_type
            ] = True  # L1: every legal ActionType the masks can express
        if not np_legal_action_type[b].any():
            # Every legal action was dropped — unrepresentable; fail loudly not NaN
            raise RuntimeError(
                f"All legal actions dropped on focus {derive_focus(game_state).name}: "
                f"{[(int(a.action_type), list(a.ids)) for a in batch_legal_actions[b]]}"
            )

        # Dedup each type's legal tokens by Card identity per token kind (copies collapse)
        for action_type, token_kind in legal_token_kinds:
            # The token kind's tokens in rank order, as their indices in the state's tokens
            np_idx = np.flatnonzero(np_token_kind[start:end] == token_kind)
            group_ids = _pile_ids(game_state, token_kind, [tokens[i][1] for i in np_idx])
            if not group_ids:
                continue

            seen: set[int] = set()
            for rank, row in enumerate(start + np_idx):
                if not np_legal[row, action_type]:
                    continue
                if group_ids[rank] in seen:
                    np_legal[row, action_type] = False
                else:
                    seen.add(group_ids[rank])
    return np_legal, np_needs_target, np_legal_action_type
