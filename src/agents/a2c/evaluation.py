import random

import torch

from src.agents.a2c.model import ActorCritic
from src.agents.a2c.model import select_action
from src.game.combat.action import ActionType
from src.game.combat.create import create_combat_state
from src.game.combat.entities import create_entity
from src.game.combat.factories import create_backflip
from src.game.combat.factories import create_dagger_throw
from src.game.combat.factories import create_defend
from src.game.combat.factories import create_strike
from src.game.combat.main import start_combat
from src.game.combat.main import step
from src.game.combat.utils import is_game_over
from src.game.combat.view import view_combat


def evaluate_blunder(model: ActorCritic, device: torch.device) -> bool:
    cs = create_combat_state()
    start_combat(cs)

    # Create a Strike, just in case we didn't draw any
    id_strike = create_entity(cs.entity_manager, create_strike())
    cs.entity_manager.id_cards_in_hand.append(id_strike)

    # Make the monster killable by 1 Strike
    monster = cs.entity_manager.entities[cs.entity_manager.id_monsters[0]]
    monster.health_current = random.randint(1, 6)

    # Set energy to 1, so that the agent only has one shot to kill the monster
    energy = cs.entity_manager.entities[cs.entity_manager.id_energy]
    energy.current = 1

    while not is_game_over(cs.entity_manager):
        combat_view = view_combat(cs)

        action = select_action(model, combat_view, greedy=True, device=device)
        if action.type == ActionType.END_TURN:
            return False

        step(cs, action)

    if monster.health_current <= 0:
        return True

    return False


def evaluate_lethal(model: ActorCritic, device: torch.device) -> bool:
    cs = create_combat_state()
    start_combat(cs)

    # Make the monster killable by 3 Strikes, but not 2
    monster = cs.entity_manager.entities[cs.entity_manager.id_monsters[0]]
    monster.health_current = random.randint(13, 18)

    # Replace hand w/ 3 Strikes and filler Defends
    cs.entity_manager.card_in_hand_ids = [
        create_entity(cs.entity_manager, create_strike()),
        create_entity(cs.entity_manager, create_strike()),
        create_entity(cs.entity_manager, create_strike()),
        create_entity(cs.entity_manager, create_defend()),
        create_entity(cs.entity_manager, create_defend()),
    ]

    while not is_game_over(cs.entity_manager):
        combat_view = view_combat(cs)

        action = select_action(model, combat_view, greedy=True, device=device)
        if action.type == ActionType.END_TURN:
            return False

        step(cs, action)

    if monster.health_current <= 0:
        return True

    return False


def evaluate_draw_first(model: ActorCritic, device: torch.device) -> bool:
    cs = create_combat_state()
    start_combat(cs)

    # Create a Backflip
    id_backflip = create_entity(cs.entity_manager, create_backflip())

    # Replace cards in the hand w/ Backflip and fillers
    cs.entity_manager.card_in_hand_ids = [
        create_entity(cs.entity_manager, create_strike()),
        create_entity(cs.entity_manager, create_strike()),
        create_entity(cs.entity_manager, create_defend()),
        create_entity(cs.entity_manager, create_defend()),
        id_backflip,
    ]

    combat_view = view_combat(cs)
    action = select_action(model, combat_view, greedy=True, device=device)

    if action.type != ActionType.SELECT_ENTITY:
        return False

    if action.target_id != id_backflip:
        return False

    return True


def evaluate_dagger_throw_over_strike(model: ActorCritic, device: torch.device) -> bool:
    cs = create_combat_state()
    start_combat(cs)

    # Create a Dagger Throw
    id_dagger_throw = create_entity(cs.entity_manager, create_dagger_throw())

    # Replace cards in the hand w/ Backflip and Strikes
    cs.entity_manager.card_in_hand_ids = [
        create_entity(cs.entity_manager, create_strike()),
        create_entity(cs.entity_manager, create_strike()),
        create_entity(cs.entity_manager, create_strike()),
        create_entity(cs.entity_manager, create_strike()),
        id_dagger_throw,
    ]

    combat_view = view_combat(cs)
    action = select_action(model, combat_view, greedy=True, device=device)

    if action.type != ActionType.SELECT_ENTITY:
        return False

    if action.target_id != id_dagger_throw:
        return False

    return True
