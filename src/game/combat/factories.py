import random

from src.game.combat.entities import Card
from src.game.combat.entities import CardName
from src.game.combat.entities import Character
from src.game.combat.entities import Effect
from src.game.combat.entities import EffectSelectionType
from src.game.combat.entities import EffectTargetType
from src.game.combat.entities import EffectType
from src.game.combat.entities import Energy
from src.game.combat.entities import Health
from src.game.combat.entities import Modifier
from src.game.combat.entities import Monster


def silent(health_current: int | None = None) -> Character:
    health_max = 50

    return Character("Silent", Health(health_max, health_current))


# TODO: improve move parametrization
def dummy(health_current: int | None = None, move_name_current: str | None = None) -> Monster:
    health_max = 30

    return Monster(
        "Dummy",
        Health(health_max, health_current),
        moves={
            "Attack": [Effect(EffectType.DEAL_DAMAGE, 10, EffectTargetType.CHARACTER)],
            "Defend": [Effect(EffectType.GAIN_BLOCK, 10, EffectTargetType.SOURCE)],
        },
        move_name_current=move_name_current,
    )


# TODO: improve move parametrization
def jaw_worm() -> Monster:
    health_max = random.randint(42, 46)

    return Monster(
        "Jaw Worm",
        Health(health_max),
        moves={
            "Chomp": [Effect(EffectType.DEAL_DAMAGE, 12, EffectTargetType.CHARACTER)],
            "Thrash": [
                Effect(EffectType.DEAL_DAMAGE, 7, EffectTargetType.CHARACTER),
                Effect(EffectType.GAIN_BLOCK, 5, EffectTargetType.SOURCE),
            ],
            "Bellow": [
                Effect(EffectType.GAIN_STR, 5, EffectTargetType.SOURCE),
                Effect(EffectType.GAIN_BLOCK, 9, EffectTargetType.SOURCE),
            ],
        },
    )


def strike() -> Card:
    cost = 1
    damage = 6

    return Card(
        CardName.STRIKE,
        cost,
        [Effect(EffectType.DEAL_DAMAGE, damage, EffectTargetType.CARD_TARGET)],
    )


def defend() -> Card:
    cost = 1
    block = 5

    return Card(
        CardName.DEFEND, cost, [Effect(EffectType.GAIN_BLOCK, block, EffectTargetType.CHARACTER)]
    )


def survivor() -> Card:
    cost = 1
    block = 8
    discard = 1

    return Card(
        CardName.SURVIVOR,
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


def neutralize() -> Card:
    cost = 0
    damage = 3
    weak = 1

    return Card(
        CardName.NEUTRALIZE,
        cost,
        [
            Effect(EffectType.DEAL_DAMAGE, damage, EffectTargetType.CARD_TARGET),
            Effect(EffectType.GAIN_WEAK, weak, EffectTargetType.CARD_TARGET),
        ],
    )


def leg_sweep() -> Card:
    cost = 2
    weak = 2
    block = 11

    return Card(
        CardName.LEG_SWEEP,
        cost,
        [
            Effect(EffectType.GAIN_BLOCK, block, EffectTargetType.CHARACTER),
            Effect(EffectType.GAIN_WEAK, weak, EffectTargetType.CARD_TARGET),
        ],
    )


def dash() -> Card:
    cost = 2
    block = 10
    damage = 10

    return Card(
        CardName.DASH,
        cost,
        [
            Effect(EffectType.GAIN_BLOCK, block, EffectTargetType.CHARACTER),
            Effect(EffectType.DEAL_DAMAGE, damage, EffectTargetType.CARD_TARGET),
        ],
    )


def acrobatics() -> Card:
    cost = 1
    draw = 3
    discard = 1

    return Card(
        CardName.ACROBATICS,
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


def dagger_throw() -> Card:
    cost = 1
    damage = 9
    draw = 1
    discard = 1

    return Card(
        CardName.DAGGER_THROW,
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


def prepared() -> Card:
    cost = 0
    draw = 1
    discard = 1

    return Card(
        CardName.PREPARED,
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


def backflip() -> Card:
    cost = 1
    block = 5
    draw = 2

    return Card(
        CardName.BACKFLIP,
        cost,
        [
            Effect(EffectType.GAIN_BLOCK, block, EffectTargetType.CHARACTER),
            Effect(EffectType.DRAW_CARD, draw),
        ],
    )


def energy(max_: int = 3, current: int = 3) -> Energy:
    return Energy(max_, current)


def weak() -> Modifier:
    return Modifier(stacks_min=0, stacks_max=999, stacks_duration=True)


def strength() -> Modifier:
    return Modifier(stacks_min=0, stacks_max=999, stacks_duration=False)
