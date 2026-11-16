import random
from typing import Any, cast

import numpy as np
import slai
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import yaml
from slai import ActionType
from slai import CardName
from slai import EffectHealthDelta
from slai import EventName
from slai import PotionName
from slai import RelicName


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

    A Potion comes from a Reward, a Shop the Character can pay or The Knowing Skull, and from none
    of them under Sozu. Alchemize is `potion_discard_wasted`'s own case.

    Parameters
    ----------
    game_state : slai.GameState
        The state to read.

    Returns
    -------
    bool
        Whether a Potion can arrive.
    """
    if any(relic.name == RelicName.Sozu for relic in game_state.relics):
        return False
    
    reward = game_state.reward
    if reward is not None and reward.potions:
        return True
    
    shop = game_state.shop
    if shop is not None and any(
        cast(int, potion.price) <= game_state.character.gold for potion in shop.potions
    ):
        return True
    
    event = game_state.event
    return event is not None and event.name == EventName.KnowingSkull and not event.consumed


def potion_discard_wasted(game_state: slai.GameState, action: slai.Action) -> bool:
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
    
    if any(relic.name == RelicName.Sozu for relic in game_state.relics):
        return True
    
    potions = game_state.potions
    if any(
        potion.name == PotionName.EntropicBrew for potion in potions if potion.id != action.ids[0]
    ):
        return False

    combat = game_state.combat
    if combat is not None and any(
        card.name == CardName.Alchemize
        for card in (*combat.card_pile_hand, *combat.card_pile_draw, *combat.card_pile_discard)
    ):
        return False
    
    return len(potions) < game_state.potion_slots_max or not _potion_can_arrive(game_state)


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
    position = next(idx for idx, option in enumerate(event.options) if option.id == id_option)
    relic_names = {relic.name for relic in game_state.relics}

    # Du-Vu Doll, Darkstone Periapt and Blue Candle give a Curse a use; Meat on the Bone, lost HP
    curse_useful = not relic_names.isdisjoint(
        {RelicName.BlueCandle, RelicName.DarkstonePeriapt, RelicName.DuVuDoll}
    )
    hp_loss_useful = RelicName.MeatOnTheBone in relic_names
    match event.name:
        # Buy one, two or three Potions, or leave, for HP from A15 on
        case EventName.TheWomanInBlue if RelicName.Sozu in relic_names:
            option_leave = event.options[3]
            leave_free = not any(
                isinstance(effect, EffectHealthDelta) for effect in option_leave.effects
            )
            return position in (1, 2) or (position == 0 and leave_free)

        # Wish for a Potion, gold or a Card, each for HP and a dearer next wish, or leave
        case EventName.KnowingSkull:
            return not hp_loss_useful and (
                (position == 0 and RelicName.Sozu in relic_names)
                or (position == 1 and RelicName.Ectoplasm in relic_names)
            )

        # Touch the face for gold and HP, trade for a face Relic, or leave
        case EventName.FaceTrader:
            return position == 0 and RelicName.Ectoplasm in relic_names and not hp_loss_useful

        # Pray for gold, desecrate for more gold and Regret, or leave
        case EventName.GoldenShrine:
            return position == 1 and RelicName.Ectoplasm in relic_names and not curse_useful

        # Accept gold and Doubt, or leave
        case EventName.TheSsssserpent:
            return position == 0 and RelicName.Ectoplasm in relic_names and not curse_useful

        # Heal to full for max HP, give the Golden Idol for gold, or leave
        case EventName.TheMoaiHead:
            return position == 0 and RelicName.MarkOfTheBloom in relic_names

        case _:
            return False
