from src.game.combat.state import Card
from src.game.combat.state import Character
from src.game.combat.state import Effect
from src.game.combat.state import EffectSelectionType
from src.game.combat.state import EffectTargetType
from src.game.combat.state import EffectType
from src.game.combat.state import Energy
from src.game.combat.state import Health
from src.game.combat.state import Monster


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
        "Strike", cost, [Effect(EffectType.DEAL_DAMAGE, damage, EffectTargetType.CARD_TARGET)]
    )


def defend() -> Card:
    cost = 1
    block = 5

    return Card("Defend", cost, [Effect(EffectType.GAIN_BLOCK, block, EffectTargetType.CHARACTER)])


def survivor() -> Card:
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
                discard,
                EffectTargetType.CARD_IN_HAND,
                EffectSelectionType.INPUT,
            ),
        ],
    )


def energy() -> Energy:
    energy = 3

    return Energy(energy)
