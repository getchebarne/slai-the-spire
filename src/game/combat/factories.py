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


def silent() -> Character:
    max_health = 70

    return Character("Silent", Health(max_health))


def dummy() -> Monster:
    max_health = 30

    return Monster("Dummy", Health(max_health))


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


def energy() -> Energy:
    energy = 3

    return Energy(energy)


def weak() -> Modifier:
    return Modifier(stacks_min=0, stacks_max=999, stacks_duration=True)
