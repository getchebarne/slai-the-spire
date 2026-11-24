"""Every hit of the walks and projections, landed on block and HP in the engine's order.

Hits land one at a time and past a kill, and nothing a hit sets off is queued.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from slai import BOOT_DAMAGE_LANDED
from slai import MAX_BOOT_DAMAGE
from slai import MAX_TORII_DAMAGE
from slai import MIN_BOOT_DAMAGE
from slai import MIN_TORII_DAMAGE
from slai import TORII_DAMAGE_LANDED
from slai import Modifier
from slai import ModifierKind

from src.rl.constants import INVINCIBLE_STACKS_OF_ASCENSION
from src.rl.types import EntityKind


@dataclass
class Combatant:
    """The Character or a Monster as the hits on it read it, moved along by each hit.

    Its Relic flags are those that act on the hits it takes: The Boot on a Monster's, Torii and
    Tungsten Rod on the Character's.
    """

    entity_kind: EntityKind  # CHARACTER or MONSTER
    health: int
    block: int
    buffer_instances: int  # HP-loss instances Buffer still absorbs
    is_intangible: bool
    invincible: int | None  # The HP Invincible still lets it lose; None without it
    has_boot: bool
    has_torii: bool
    has_tungsten_rod: bool
    revives: list[int]  # The HP each revive left restores, the next one first

    damage: int = 0  # Dealt so far: what landed on block and HP
    health_lost: int = 0
    is_lethal: bool = False


def make_combatant(
    entity_kind: EntityKind,
    health: int,
    block: int,
    stacks_of_modifier_kind: dict[ModifierKind, int],
    has_boot: bool = False,
    has_torii: bool = False,
    has_tungsten_rod: bool = False,
    revives: Sequence[int] = (),
) -> Combatant:
    """Make a Combatant, its Buffer, Intangible and Invincible read from its Modifiers.

    Parameters
    ----------
    entity_kind : EntityKind
        CHARACTER or MONSTER.
    health : int
        Its health.
    block : int
        Its block, as the hits meet it.
    stacks_of_modifier_kind : dict[ModifierKind, int]
        Its Modifiers: each kind's stacks.
    has_boot : bool, default False
        Whether The Boot acts on the hits it takes: a Monster's, while it is held.
    has_torii : bool, default False
        Whether Torii acts on them: the Character's, while it is held.
    has_tungsten_rod : bool, default False
        Whether Tungsten Rod acts on them: the Character's, while it is held.
    revives : Sequence[int], default ()
        The HP each revive left restores, the next one first.

    Returns
    -------
    Combatant
        Its side of the hits, nothing dealt yet.
    """
    return Combatant(
        entity_kind=entity_kind,
        health=health,
        block=block,
        buffer_instances=stacks_of_modifier_kind.get(ModifierKind.Buffer, 0),
        is_intangible=ModifierKind.Intangible in stacks_of_modifier_kind,
        invincible=stacks_of_modifier_kind.get(ModifierKind.Invincible),
        has_boot=has_boot,
        has_torii=has_torii,
        has_tungsten_rod=has_tungsten_rod,
        revives=list(revives),
    )


def get_stacks_of_modifier_kind(modifiers: list[Modifier]) -> dict[ModifierKind, int]:
    """Return each kind's stacks among a combatant's Modifiers.

    Parameters
    ----------
    modifiers : list[Modifier]
        The Character's or a Monster's Modifiers.

    Returns
    -------
    dict[ModifierKind, int]
        The stacks of each kind it holds.
    """
    return {modifier.kind: modifier.stacks for modifier in modifiers}


def get_poison_tick(combatant_monster: Combatant, poison_monster: int, ascension: int) -> int:
    """Refill Invincible's cap at a Monster's turn start, then land a tick of `poison_monster`.

    Parameters
    ----------
    combatant_monster : Combatant
        The Monster, moved along in place.
    poison_monster : int
        The Poison stacks.
    ascension : int
        The run's Ascension, which sets Invincible's cap.

    Returns
    -------
    int
        The HP the tick takes.
    """
    # Invincible's cap refills ahead of its holder's tick
    if combatant_monster.invincible is not None:
        combatant_monster.invincible = INVINCIBLE_STACKS_OF_ASCENSION[ascension]
    health_lost_before = combatant_monster.health_lost
    lose_health(combatant_monster, poison_monster, None)

    return combatant_monster.health_lost - health_lost_before


def lose_health(combatant: Combatant, amount: int, entity_kind_source: EntityKind | None) -> int:
    """Take an HP loss off `combatant`, as `process_effect_health_delta` takes one.

    Parameters
    ----------
    combatant : Combatant
        Who loses it, moved along in place.
    amount : int
        The HP it would take.
    entity_kind_source : EntityKind | None
        Its source's Entity kind: CARD, MONSTER, or None for a loss with no source.

    Returns
    -------
    int
        What it deals: its amount after Intangible's clamp.
    """
    # A Card's hit on a Monster: its Intangible was clamped before block, ahead of The Boot's lift
    is_card_hit_on_monster = (
        entity_kind_source == EntityKind.CARD and combatant.entity_kind == EntityKind.MONSTER
    )

    # Intangible clamps the instance
    if combatant.is_intangible and not is_card_hit_on_monster:
        amount = min(amount, 1)
    damage = amount

    # Invincible: its holder loses no more than the cap has left, and the cap drops by the loss
    if combatant.invincible is not None:
        amount = min(amount, combatant.invincible)
        combatant.invincible -= amount

    # Buffer absorbs it whole
    if amount > 0 and combatant.buffer_instances > 0:
        combatant.buffer_instances -= 1

        return damage

    # Tungsten Rod takes 1 off
    if combatant.entity_kind == EntityKind.CHARACTER and amount > 0 and combatant.has_tungsten_rod:
        amount -= 1

    # Accumulate lost health
    combatant.health -= amount
    combatant.health_lost += amount

    # Lethal check
    if combatant.health <= 0:
        if combatant.revives:
            combatant.health = combatant.revives.pop(0)
        else:
            combatant.is_lethal = True

    return damage


def deal_damage(combatant: Combatant, amount: int, entity_kind_source: EntityKind | None) -> None:
    """Land a hit on `combatant`, as `process_effect_damage_deal` lands one.

    Parameters
    ----------
    combatant : Combatant
        Who it hits, moved along in place.
    amount : int
        The hit.
    entity_kind_source : EntityKind | None
        Its source's Entity kind: CARD, MONSTER, or None for a hit with no source.
    """
    if combatant.is_intangible:
        amount = min(amount, 1)

    # Accumulate damage taken
    combatant.damage += amount

    # Calculate damage over block, set new current block
    damage_over_block = max(amount - combatant.block, 0)
    combatant.block = max(combatant.block - amount, 0)

    # Apply The Boot's lift
    if (
        entity_kind_source == EntityKind.CARD
        and combatant.entity_kind == EntityKind.MONSTER
        and combatant.has_boot
        and MIN_BOOT_DAMAGE <= damage_over_block <= MAX_BOOT_DAMAGE
    ):
        damage_over_block = BOOT_DAMAGE_LANDED

    # Apply Torii reduction
    if (
        entity_kind_source == EntityKind.MONSTER
        and combatant.entity_kind == EntityKind.CHARACTER
        and combatant.has_torii
        and MIN_TORII_DAMAGE <= damage_over_block <= MAX_TORII_DAMAGE
    ):
        damage_over_block = TORII_DAMAGE_LANDED

    # Accumulate lost health if any
    if damage_over_block > 0:
        lose_health(combatant, damage_over_block, entity_kind_source)
