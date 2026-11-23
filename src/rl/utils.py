import random
from typing import Any, cast

import numpy as np
import slai
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import yaml
from slai import ACT_FINAL
from slai import ActionType
from slai import CardName
from slai import EffectHealthDelta
from slai import EffectRoomEnter
from slai import EventName
from slai import PotionName
from slai import RelicName

from src.rl.constants import MAP_HEIGHT
from src.rl.tokens import get_burning_path
from src.rl.tokens import get_character
from src.rl.tokens import get_entities
from src.rl.tokens import get_room_burning
from src.rl.tokens import get_rooms
from src.rl.tokens import get_y_x_current
from src.rl.tokens import relic_held


# The Relics that give a Curse a use
_RELICS_CURSE_USEFUL = frozenset(
    {RelicName.BlueCandle, RelicName.DarkstonePeriapt, RelicName.DuVuDoll}
)


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Parse integer values
    if "num_iterations" in config:
        config["num_iterations"] = int(config["num_iterations"])

    return config


def set_rng_seeds(seed: int) -> None:
    """Seed the global generators of random, numpy and torch.

    Parameters
    ----------
    seed : int
        The seed, taken modulo 2**32 for numpy.
    """
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def init_optimizer(optimizer_name: str, model: nn.Module, **kwargs) -> torch.optim.Optimizer:
    return getattr(torch.optim, optimizer_name)(**kwargs, params=model.parameters())


def parent_alive() -> bool:
    """Return whether the process that started this one is still alive.

    Returns
    -------
    bool
        Whether the parent process is alive; False when this process has none.
    """
    parent = mp.parent_process()

    return parent is not None and parent.is_alive()


def _potion_can_arrive(game_state: slai.GameState) -> bool:
    """Return whether a free belt slot could take a Potion from this state.

    A Potion comes from a Reward, a Shop the Character can pay or The Knowing Skull. Sozu and
    Alchemize are `_potion_discard_wasted`'s own cases.

    Parameters
    ----------
    game_state : slai.GameState
        The state to read.

    Returns
    -------
    bool
        Whether a Potion can arrive.
    """
    reward = game_state.reward
    if reward is not None and reward.id_potions:
        return True
    
    shop = game_state.shop
    if shop is not None and any(
        cast(int, potion.price) <= get_character(game_state).gold
        for potion in get_entities(game_state, shop.id_potions, slai.Potion)
    ):
        return True
    
    event = game_state.event

    return event is not None and event.name == EventName.KnowingSkull and not event.consumed


def _potion_discard_wasted(game_state: slai.GameState, action: slai.Action) -> bool:
    """Return whether `action` discards a Potion while nothing needs the room.

    No Potion can arrive under Sozu. An Entropic Brew held in another slot makes the discard pay as
    a reroll. An Alchemize in a Combat pile can add two Potions at once (Burst, a Duplication
    Potion) or be played from the draw pile (Mayhem, Distilled Chaos), each with no decision in
    between, so the room can pay with a free slot or nothing in hand. Otherwise the room only pays
    on a full belt, where a Potion can arrive.

    Parameters
    ----------
    game_state : slai.GameState
        The state `action` is legal in.
    action : slai.Action
        One legal action.

    Returns
    -------
    bool
        True for a Potion discard under Sozu, or with no Entropic Brew in another slot, no
        Alchemize in the hand, draw pile or discard pile, and either a free belt slot or no Potion
        that can arrive.
    """
    if action.action_type is not ActionType.PotionDiscard:
        return False
    
    if relic_held(game_state, RelicName.Sozu):
        return True
    
    potions = get_entities(game_state, game_state.id_potions, slai.Potion)
    if any(
        potion.name == PotionName.EntropicBrew for potion in potions if potion.id != action.ids[0]
    ):
        return False

    combat = game_state.combat
    if combat is not None and any(
        card.name == CardName.Alchemize
        for card in get_entities(
            game_state,
            (*combat.id_card_pile_hand, *combat.id_card_pile_draw, *combat.id_card_pile_discard),
            slai.Card,
        )
    ):
        return False
    
    return len(potions) < game_state.potion_slots_max or not _potion_can_arrive(game_state)


def _potion_take_wasted(game_state: slai.GameState, action: slai.Action) -> bool:
    """Return whether `action` takes a Reward's Potion under Sozu, which obtains nothing.

    The Shop's Potions need no such check: the engine doesn't offer them under Sozu.

    Parameters
    ----------
    game_state : slai.GameState
        The state `action` is legal in.
    action : slai.Action
        One legal action.

    Returns
    -------
    bool
        True for a Potion take while Sozu is held.
    """
    return action.action_type is ActionType.RewardTakePotion and relic_held(
        game_state, RelicName.Sozu
    )


def curse_useful(game_state: slai.GameState) -> bool:
    """Return whether a held Relic gives a Curse a use.

    Du-Vu Doll and Darkstone Periapt pay for each Curse, and Blue Candle makes one playable.

    Parameters
    ----------
    game_state : slai.GameState
        The state to read.

    Returns
    -------
    bool
        Whether Blue Candle, Darkstone Periapt or Du-Vu Doll is held.
    """
    return not _RELICS_CURSE_USEFUL.isdisjoint(
        relic.name for relic in get_entities(game_state, game_state.id_relics, slai.Relic)
    )


def event_option_wasted(game_state: slai.GameState, id_option: int) -> bool:
    """Return whether selecting the Event's option pays for a gain that a held Relic voids.

    Sozu voids Potions, Ectoplasm gold and Mark of the Bloom healing. Each option below pays HP,
    max HP, gold or a Curse for such a gain alone, while leaving, or a cheaper option, is open.

    Parameters
    ----------
    game_state : slai.GameState
        The state to read, at an Event.
    id_option : int
        The id of one of the Event's options.

    Returns
    -------
    bool
        True under Sozu for The Woman in Blue's purchases but the cheapest, that one too when
        leaving is free, and for The Knowing Skull's Potion wish; under Ectoplasm for The Knowing
        Skull's gold wish, Face Trader's touch, Golden Shrine's desecration and The Ssssserpent's
        offer; under Mark of the Bloom for The Moai Head's heal. The Curse options stay open with
        Du-Vu Doll, Darkstone Periapt or Blue Candle held, the HP-loss ones with Meat on the Bone.
    """
    event = cast(slai.Event, game_state.event)

    # The options are in catalog order, the order each case below lists them in
    idx_option = event.id_event_options.index(id_option)
    relic_names = {
        relic.name for relic in get_entities(game_state, game_state.id_relics, slai.Relic)
    }

    # Meat on the Bone gives lost HP a use
    hp_loss_useful = RelicName.MeatOnTheBone in relic_names
    match event.name:
        # Buy one, two or three Potions, or leave, for HP from A15 on
        case EventName.TheWomanInBlue if RelicName.Sozu in relic_names:
            (option_leave,) = get_entities(
                game_state, [event.id_event_options[3]], slai.EventOption
            )
            leave_free = not any(
                isinstance(effect, EffectHealthDelta) for effect in option_leave.effects
            )

            return idx_option in (1, 2) or (idx_option == 0 and leave_free)

        # Wish for a Potion, gold or a Card, each for HP and a dearer next wish, or leave
        case EventName.KnowingSkull:
            return not hp_loss_useful and (
                (idx_option == 0 and RelicName.Sozu in relic_names)
                or (idx_option == 1 and RelicName.Ectoplasm in relic_names)
            )

        # Touch the face for gold and HP, trade for a face Relic, or leave
        case EventName.FaceTrader:
            return idx_option == 0 and RelicName.Ectoplasm in relic_names and not hp_loss_useful

        # Pray for gold, desecrate for more gold and Regret, or leave
        case EventName.GoldenShrine:
            return (
                idx_option == 1
                and RelicName.Ectoplasm in relic_names
                and not curse_useful(game_state)
            )

        # Accept gold and Doubt, or leave
        case EventName.TheSsssserpent:
            return (
                idx_option == 0
                and RelicName.Ectoplasm in relic_names
                and not curse_useful(game_state)
            )

        # Heal to full for max HP, give the Golden Idol for gold, or leave
        case EventName.TheMoaiHead:
            return idx_option == 0 and RelicName.MarkOfTheBloom in relic_names

        case _:
            return False


def action_masked(game_state: slai.GameState, action: slai.Action) -> bool:
    """Return whether the masks remove `action`, an engine-legal action the policy never needs.

    The encoder's masks, the scripted fold and the Potion `usable` column all read this one check,
    so a new mask also changes that model input. The Card dedup is the encoder's own.

    Parameters
    ----------
    game_state : slai.GameState
        The state `action` is legal in.
    action : slai.Action
        One legal action.

    Returns
    -------
    bool
        True for a wasted Potion discard, a Potion take under Sozu, an Event option a held Relic
        voids and an action that gives up a key.
    """
    return (
        _potion_discard_wasted(game_state, action)
        or _potion_take_wasted(game_state, action)
        or (
            action.action_type is ActionType.EventOptionSelect
            and event_option_wasted(game_state, action.ids[0])
        )
        or _forfeits_key(game_state, action)
    )


def _misses_room_burning(game_state: slai.GameState, action: slai.Action) -> bool:
    """Return whether a RoomSelect leaves the burning Room's path.

    Parameters
    ----------
    game_state : slai.GameState
        The state `action` is legal in.
    action : slai.Action
        A legal RoomSelect.

    Returns
    -------
    bool
        True when the selected Room is off `get_burning_path` while another Room in its row is on
        it.
    """
    id_room = action.ids[0]
    (room_selected,) = get_entities(game_state, [id_room], slai.Room)
    burning_path = get_burning_path(game_state)
    
    return (room_selected.y, room_selected.x) not in burning_path and any(
        y == room_selected.y for y, _ in burning_path
    )


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
    (potion,) = get_entities(game_state, [id_potion], slai.Potion)
    if potion.name != PotionName.SmokeBomb:
        return False

    room_burning = get_room_burning(get_rooms(game_state))
    y_current, x_current = get_y_x_current(game_state.map)
    
    return (
        room_burning is not None
        and y_current == room_burning.y
        and x_current == room_burning.x
    )


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
        True for a RoomExit at a Chest or while `id_relic_linked` is set, and for taking the linked
        Relic.
    """
    # Row 8 is always a Chest
    y_current, _ = get_y_x_current(game_state.map)
    if y_current is None or y_current < 8:
        return False

    # An opened Chest closes for its Reward
    reward = game_state.reward
    if reward is None:
        return game_state.chest is not None

    # Took the linked Relic instead of the key
    id_relic_linked = reward.id_relic_linked
    
    return id_relic_linked is not None and (
        action.action_type is ActionType.RoomExit or action.ids[0] == id_relic_linked
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
    (option,) = get_entities(game_state, [id_option], slai.EventOption)

    return any(isinstance(effect, EffectRoomEnter) for effect in option.effects)


def _forfeits_key(game_state: slai.GameState, action: slai.Action) -> bool:
    """Return whether `action` gives up a key.

    Only Act 3 can lose a key.

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
        game_state.act != ACT_FINAL - 1
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
            y_current, _ = get_y_x_current(game_state.map)

            return not game_state.key_ruby and y_current == MAP_HEIGHT - 1
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
