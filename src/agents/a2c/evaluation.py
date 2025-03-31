import random

from src.agents.a2c.model import ActorCritic
from src.agents.a2c.model import select_action
from src.game.combat.action import ActionType
from src.game.combat.effect import Effect
from src.game.combat.effect import EffectTargetType
from src.game.combat.effect import EffectType
from src.game.combat.entities import EntityManager
from src.game.combat.entities import MonsterMove
from src.game.combat.entities import create_entity
from src.game.combat.factories import create_backflip
from src.game.combat.factories import create_dagger_throw
from src.game.combat.factories import create_dash
from src.game.combat.factories import create_defend
from src.game.combat.factories import create_dummy
from src.game.combat.factories import create_energy
from src.game.combat.factories import create_silent
from src.game.combat.factories import create_strike
from src.game.combat.main import step
from src.game.combat.state import CombatState
from src.game.combat.utils import is_game_over
from src.game.combat.view import view_combat


# TODO: improve
def _create_combat_state(
    health_current_char: int | None = None,
    health_current_dummy: int | None = None,
    move_current: MonsterMove | None = None,
    energy_current: int | None = None,
) -> CombatState:
    # Create entities
    entity_manager = EntityManager([])
    id_character = create_entity(
        entity_manager, create_silent(health_current_char, health_current_char)
    )
    id_monster = create_entity(
        entity_manager, create_dummy(health_current_dummy, health_current_dummy, move_current)
    )
    id_energy = create_entity(entity_manager, create_energy(3, energy_current))

    entity_manager.id_character = id_character
    entity_manager.id_monsters = [id_monster]
    entity_manager.id_energy = id_energy

    return CombatState(entity_manager, [])


def evaluate_blunder(model: ActorCritic) -> bool:
    cs = _create_combat_state(
        health_current_char=random.randint(1, 6),
        health_current_dummy=random.randint(1, 6),
        energy_current=1,
        move_current=random.choice(
            [
                MonsterMove(
                    "Attack", [Effect(EffectType.DEAL_DAMAGE, 12, EffectTargetType.CHARACTER)]
                ),
                MonsterMove(
                    "Defend", [Effect(EffectType.GAIN_BLOCK, 12, EffectTargetType.SOURCE)]
                ),
            ]
        ),
    )
    id_strike = create_entity(cs.entity_manager, create_strike())
    id_defend = create_entity(cs.entity_manager, create_defend())
    cs.entity_manager.id_cards_in_hand = [id_strike, id_defend]

    combat_view = view_combat(cs)
    action = select_action(model, combat_view, greedy=True)

    if action.type != ActionType.SELECT_ENTITY:
        return False

    if action.target_id != id_strike:
        return False

    return True


def evaluate_lethal(model: ActorCritic) -> bool:
    cs = _create_combat_state(
        health_current_char=random.randint(1, 6),
        health_current_dummy=random.randint(13, 18),
        energy_current=3,
        move_current=random.choice(
            [
                MonsterMove(
                    "Attack", [Effect(EffectType.DEAL_DAMAGE, 12, EffectTargetType.CHARACTER)]
                ),
                MonsterMove(
                    "Defend", [Effect(EffectType.GAIN_BLOCK, 12, EffectTargetType.SOURCE)]
                ),
            ]
        ),
    )
    cs.entity_manager.card_in_hand_ids = [
        create_entity(cs.entity_manager, create_strike()),
        create_entity(cs.entity_manager, create_strike()),
        create_entity(cs.entity_manager, create_strike()),
        create_entity(cs.entity_manager, create_defend()),
    ]

    while not is_game_over(cs.entity_manager):
        # Get combat view
        combat_view = view_combat(cs)

        # Get action from agent
        action = select_action(model, combat_view, greedy=True)
        if action.type == ActionType.END_TURN:
            return False

        # Game step
        step(cs, action)

    return True


def evaluate_draw_first(model: ActorCritic) -> bool:
    cs = _create_combat_state(
        health_current_char=random.randint(1, 6),
        health_current_dummy=random.randint(20, 25),
        energy_current=3,
        move_current=MonsterMove(
            "Attack", [Effect(EffectType.DEAL_DAMAGE, 12, EffectTargetType.CHARACTER)]
        ),
    )
    id_backflip = create_entity(cs.entity_manager, create_backflip())
    cs.entity_manager.card_in_hand_ids = [
        create_entity(cs.entity_manager, create_strike()),
        create_entity(cs.entity_manager, create_strike()),
        create_entity(cs.entity_manager, create_strike()),
        create_entity(cs.entity_manager, create_defend()),
        id_backflip,
    ]
    cs.entity_manager.card_in_draw_pile_ids = [
        create_entity(cs.entity_manager, create_strike()),
        create_entity(cs.entity_manager, create_dash()),
        create_entity(cs.entity_manager, create_defend()),
    ]

    combat_view = view_combat(cs)
    action = select_action(model, combat_view, greedy=True)

    if action.type != ActionType.SELECT_ENTITY:
        return False

    if action.target_id != id_backflip:
        return False

    return True


def evaluate_dagger_throw_over_strike(model: ActorCritic) -> bool:
    cs = _create_combat_state(
        health_current_char=random.randint(1, 6),
        health_current_dummy=random.randint(20, 25),
        energy_current=1,
        move_current=random.choice(
            [
                MonsterMove(
                    "Attack", [Effect(EffectType.DEAL_DAMAGE, 12, EffectTargetType.CHARACTER)]
                ),
                MonsterMove(
                    "Defend", [Effect(EffectType.GAIN_BLOCK, 12, EffectTargetType.SOURCE)]
                ),
            ]
        ),
    )
    id_dagger_throw = create_entity(cs.entity_manager, create_dagger_throw())
    cs.entity_manager.card_in_hand_ids = [
        create_entity(cs.entity_manager, create_strike()),
        create_entity(cs.entity_manager, create_strike()),
        create_entity(cs.entity_manager, create_strike()),
        id_dagger_throw,
    ]
    cs.entity_manager.card_in_draw_pile_ids = [
        create_entity(cs.entity_manager, create_strike()),
        create_entity(cs.entity_manager, create_dash()),
        create_entity(cs.entity_manager, create_defend()),
    ]

    combat_view = view_combat(cs)
    action = select_action(model, combat_view, greedy=True)

    if action.type != ActionType.SELECT_ENTITY:
        return False

    if action.target_id != id_dagger_throw:
        return False

    return True
