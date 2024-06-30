from src.game.combat.context import Card
from src.game.combat.context import Character
from src.game.combat.context import Effect
from src.game.combat.context import EffectTargetType
from src.game.combat.context import EffectType
from src.game.combat.context import Energy
from src.game.combat.context import Health
from src.game.combat.context import Monster


def silent() -> Character:
    max_health = 70

    return Character("Silent", Health(max_health))


def dummy() -> Monster:
    max_health = 30

    return Character("Dummy", Health(max_health))


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


def energy() -> Energy:
    energy = 3

    return Energy(energy)
