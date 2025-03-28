import random

from src.game.combat.effect import Effect
from src.game.combat.effect import EffectSelectionType
from src.game.combat.effect import EffectTargetType
from src.game.combat.effect import EffectType
from src.game.combat.entities import Card
from src.game.combat.entities import Character
from src.game.combat.entities import Energy
from src.game.combat.entities import Monster
from src.game.combat.entities import MonsterMove


def create_silent(health_current: int, health_max: int) -> Character:
    return Character("Silent", health_current=health_current, health_max=health_max)


def create_dummy(health_current: int, health_max: int) -> Monster:
    return Monster("Dummy", health_current=health_current, health_max=health_max)


def create_jaw_worm(move_current: MonsterMove) -> Monster:
    health_max = random.randint(42, 46)
    health_current = health_max

    return Monster(
        "Jaw Worm", health_current=health_current, health_max=health_max, move_current=move_current
    )


def create_strike() -> Card:
    cost = 1
    damage = 6

    return Card(
        "Strike",
        cost,
        [Effect(EffectType.DEAL_DAMAGE, damage, EffectTargetType.CARD_TARGET)],
    )


def create_defend() -> Card:
    cost = 1
    block = 5

    return Card("Defend", cost, [Effect(EffectType.GAIN_BLOCK, block, EffectTargetType.CHARACTER)])


def create_survivor() -> Card:
    cost = 1
    block = 8
    discard = 1

    return Card(
        "Survivor",
        cost,
        [
            Effect(EffectType.GAIN_BLOCK, block, EffectTargetType.CHARACTER),
            Effect(
                EffectType.DISCARD,
                discard,  # TODO: this should be part of the selection type
                EffectTargetType.CARD_IN_HAND,
                EffectSelectionType.INPUT,
            ),
        ],
    )


def create_neutralize() -> Card:
    cost = 0
    damage = 3
    weak = 1

    return Card(
        "Neutralize",
        cost,
        [
            Effect(EffectType.DEAL_DAMAGE, damage, EffectTargetType.CARD_TARGET),
            Effect(EffectType.GAIN_WEAK, weak, EffectTargetType.CARD_TARGET),
        ],
    )


def create_leg_sweep() -> Card:
    cost = 2
    weak = 2
    block = 11

    return Card(
        "Leg Sweep",
        cost,
        [
            Effect(EffectType.GAIN_BLOCK, block, EffectTargetType.CHARACTER),
            Effect(EffectType.GAIN_WEAK, weak, EffectTargetType.CARD_TARGET),
        ],
    )


def create_dash() -> Card:
    cost = 2
    block = 10
    damage = 10

    return Card(
        "Dash",
        cost,
        [
            Effect(EffectType.GAIN_BLOCK, block, EffectTargetType.CHARACTER),
            Effect(EffectType.DEAL_DAMAGE, damage, EffectTargetType.CARD_TARGET),
        ],
    )


def create_acrobatics() -> Card:
    cost = 1
    draw = 3
    discard = 1

    return Card(
        "Acrobatics",
        cost,
        [
            Effect(EffectType.DRAW_CARD, draw),
            Effect(
                EffectType.DISCARD,
                discard,
                EffectTargetType.CARD_IN_HAND,
                EffectSelectionType.INPUT,
            ),
        ],
    )


def create_dagger_throw() -> Card:
    cost = 1
    damage = 9
    draw = 1
    discard = 1

    return Card(
        "Dagger Throw",
        cost,
        [
            Effect(EffectType.DEAL_DAMAGE, damage, EffectTargetType.CARD_TARGET),
            Effect(EffectType.DRAW_CARD, draw),
            Effect(
                EffectType.DISCARD,
                discard,
                EffectTargetType.CARD_IN_HAND,
                EffectSelectionType.INPUT,
            ),
        ],
    )


def create_backflip() -> Card:
    cost = 1
    block = 5
    draw = 2

    return Card(
        "Backflip",
        cost,
        [
            Effect(EffectType.GAIN_BLOCK, block, EffectTargetType.CHARACTER),
            Effect(EffectType.DRAW_CARD, draw),
        ],
    )


def create_energy(max_: int, current: int) -> Energy:
    return Energy(max_, current)
