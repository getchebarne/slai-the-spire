"""What ending the Character's turn now would cost it.

Walks what the engine runs from the turn end to the last Monster's move, on the state as it stands:
block gained at the turn end, Cards that play themselves, Poison ticks and the deaths they set off
ahead of the moves, and each hit through the rules that change what it costs.
"""

from dataclasses import dataclass
from typing import cast

from slai import AmountRelative
from slai import CardName
from slai import CardPile
from slai import EffectCardDiscoverSelect
from slai import EffectHealthDelta
from slai import GameState
from slai import Intent
from slai import ModifierKind
from slai import Monster
from slai import MonsterName
from slai import PotionName
from slai import RelicName

from src.rl.constants import INVINCIBLE_STACKS_OF_ASCENSION


# The engine's hardcoded amounts, which GameEnv doesn't expose: `process_card_play`,
# `process_effect_turn_end_character` and `process_effect_move_execute`
_ORICHALCUM_BLOCK = 6
_BURN_DAMAGE = 2
_BURN_UPGRADED_DAMAGE = 4
_DECAY_DAMAGE = 2
_EXPLOSIVE_DAMAGE = 30

# A Monster hit leaving this much past block lands 1 under Torii
_TORII_MIN = 2
_TORII_MAX = 5


@dataclass
class _Walk:
    """The Character's side of the walk, moved along by each hit."""

    health: int
    block: int
    buffer: int  # HP-loss instances Buffer still absorbs
    intangible: bool
    torii: bool
    tungsten_rod: bool
    revives: list[int]  # The HP each revive left restores, the next one first
    damage: int = 0  # Dealt so far: what landed on block and HP
    health_lost: int = 0
    is_lethal: bool = False


@dataclass
class Incoming:
    """What ending the Character's turn now would cost it, overall and per Monster.

    Both per-Monster dicts hold every Monster, by id.
    """

    damage: int  # What lands on block and HP
    health_lost: int
    is_lethal: bool
    health_at_monsters: int  # The HP left when the Monsters' moves start, the revives' included
    damage_of_monster: dict[int, int]
    health_lost_of_monster: dict[int, int]


def get_poison_tick(
    poison: int, intangible: bool, stacks_of_modifier_kind: dict, ascension: int
) -> int:
    """Return the HP a tick of `poison` takes from a Monster holding those Modifiers.

    Parameters
    ----------
    poison : int
        The Poison stacks.
    intangible : bool
        Whether Intangible covers the tick.
    stacks_of_modifier_kind : dict
        The Monster's Modifiers: each kind's stacks.
    ascension : int
        The run's Ascension, which sets Invincible's cap.

    Returns
    -------
    int
        The HP the tick takes.
    """
    if intangible:
        return min(poison, 1)

    # Invincible's cap refills ahead of its holder's tick
    if ModifierKind.Invincible in stacks_of_modifier_kind:
        return min(poison, INVINCIBLE_STACKS_OF_ASCENSION[ascension])
    return poison


def _lose_health(walk: _Walk, amount: int) -> None:
    # Intangible clamps the instance, Buffer absorbs it whole, and Tungsten Rod takes 1 off
    if walk.intangible:
        amount = min(amount, 1)
    if walk.buffer > 0:
        walk.buffer -= 1
        return
    if walk.tungsten_rod:
        amount -= 1
    walk.health -= amount
    walk.health_lost += amount
    if walk.health <= 0:
        if walk.revives:
            walk.health = walk.revives.pop(0)
        else:
            walk.is_lethal = True


def _deal_damage(walk: _Walk, amount: int, from_monster: bool) -> None:
    if walk.intangible:
        amount = min(amount, 1)
    walk.damage += amount
    past_block = max(amount - walk.block, 0)
    walk.block = max(walk.block - amount, 0)
    if from_monster and walk.torii and _TORII_MIN <= past_block <= _TORII_MAX:
        past_block = 1
    if past_block > 0:
        _lose_health(walk, past_block)


def _resolve_deaths(
    monsters: list[Monster],
    stacks_of_monster: dict[int, dict],
    health_of_monster: dict[int, int],
    exploded: set[int],
) -> None:
    """Resolve what the Monsters' deaths set off, ahead of the next move.

    Parameters
    ----------
    monsters : list[Monster]
        The Monsters.
    stacks_of_monster : dict[int, dict]
        Each Monster's Modifiers, by id: each kind's stacks.
    health_of_monster : dict[int, int]
        Each Monster's health by id, with the block Barricade keeps; 0 or less for one that won't
        move. Moved along in place.
    exploded : set[int]
        The ids whose Corpse Explosion went off. Moved along in place.
    """
    # Corpse Explosion: a death deals its max health per stack to the others, which can chain
    for _ in monsters:
        for monster in monsters:
            corpse_explosion = stacks_of_monster[monster.id].get(ModifierKind.CorpseExplosion, 0)
            if (
                corpse_explosion > 0
                and health_of_monster[monster.id] <= 0
                and not monster.half_dead
                and monster.id not in exploded
            ):
                exploded.add(monster.id)
                for other in monsters:
                    if other.id != monster.id:
                        health_of_monster[other.id] -= monster.health_max * corpse_explosion

    # The Awakened One's final death sends its Cultists running
    for monster in monsters:
        if (
            monster.name == MonsterName.AwakenedOne
            and not monster.half_dead
            and ModifierKind.Unawakened not in stacks_of_monster[monster.id]
            and health_of_monster[monster.id] <= 0
        ):
            for other in monsters:
                if other.name == MonsterName.Cultist:
                    health_of_monster[other.id] = 0

    # A leader's death takes its Minions along, or sends them running
    dead = [monster for monster in monsters if health_of_monster[monster.id] <= 0]
    alive = [monster for monster in monsters if health_of_monster[monster.id] > 0]
    if any(ModifierKind.Minion not in stacks_of_monster[monster.id] for monster in dead) and all(
        ModifierKind.Minion in stacks_of_monster[monster.id] for monster in alive
    ):
        for monster in alive:
            health_of_monster[monster.id] = 0


def _walk_monster_move(
    walk: _Walk, monster: Monster, stacks: dict, health: int, thorns: int
) -> int:
    """Walk the Monster's move.

    Parameters
    ----------
    walk : _Walk
        The walk, moved along in place.
    monster : Monster
        The Monster, with its intent visible.
    stacks : dict
        The Monster's Modifiers: each kind's stacks.
    health : int
        Its health at its move, with the block Barricade keeps.
    thorns : int
        The Character's Thorns stacks.

    Returns
    -------
    int
        Its health after the move, likewise; 0 or less for one that's dead.
    """

    # A dead Monster skips its move, and a half-dead one hits nothing
    if monster.half_dead or health <= 0:
        return health

    # HP lost ahead of the move that crosses Split or Mode Shift replaces the move
    health_lost = monster.health - health
    if health_lost > 0 and (
        (ModifierKind.Splittable in stacks and health <= monster.health_max // 2)
        or 0 < stacks.get(ModifierKind.ModeShift, 0) <= health_lost
    ):
        return health

    # Shifting: HP lost comes off its Strength until its turn end
    intent = cast(Intent, monster.intent)
    hit = intent.damage or 0
    if ModifierKind.Shifting in stacks:
        hit = max(hit - health_lost, 0)

    # Thorns can kill it mid-move, which cancels its remaining hits; its Intangible clamps each
    # reflect
    if ModifierKind.Intangible in stacks:
        thorns = min(thorns, 1)
    for _ in range(intent.instances or 0):
        if health <= 0:
            break
        _deal_damage(walk, hit, from_monster=True)
        health -= thorns

    # Explosive's last stack blows up behind the move, with no source, and takes the Monster along
    if stacks.get(ModifierKind.Explosive, 0) == 1:
        _deal_damage(walk, _EXPLOSIVE_DAMAGE, from_monster=False)
        return 0

    return health


def _get_revives(game_state: GameState) -> list[int]:
    """Return the HP each revive restores, in the order they fire.

    Parameters
    ----------
    game_state : GameState
        The state to read.

    Returns
    -------
    list[int]
        Each Fairy in a Bottle's heal, then Lizard Tail's half max HP if it's unspent.
    """
    health_max = game_state.character.health_max
    revives = []
    for potion in game_state.potions:
        if potion.name == PotionName.Fairy:
            amount = cast(AmountRelative, cast(EffectHealthDelta, potion.effects[0]).amount)
            revives.append(max(health_max * amount.numerator // amount.denominator, 1))
    if any(
        relic.name == RelicName.LizardTail and not relic.used_up for relic in game_state.relics
    ):
        revives.append(max(health_max // 2, 1))
    return revives


def project_incoming(game_state: GameState) -> Incoming | None:
    """Walk what ending the Character's turn now would cost it.

    Parameters
    ----------
    game_state : GameState
        The state to read.

    Returns
    -------
    Incoming | None
        What the walk found; None outside Combat, or when Runic Dome hides the intents.
    """
    combat = game_state.combat
    if combat is None:
        return None

    monsters = combat.monsters
    if any(monster.intent is None for monster in monsters):
        return None

    character = game_state.character
    stacks_of_modifier_kind = {modifier.kind: modifier.stacks for modifier in character.modifiers}
    relic_names = {relic.name for relic in game_state.relics}
    walk = _Walk(
        health=character.health,
        block=character.block,
        buffer=stacks_of_modifier_kind.get(ModifierKind.Buffer, 0),
        intangible=ModifierKind.Intangible in stacks_of_modifier_kind,
        torii=RelicName.Torii in relic_names,
        tungsten_rod=RelicName.TungstenRod in relic_names,
        revives=[] if RelicName.MarkOfTheBloom in relic_names else _get_revives(game_state),
    )

    # A selection inside the turn end comes after its first pass and the heal, but for Nilry's
    # Codex in the first pass, with Orichalcum already landed
    effect_pending = game_state.effect_pending
    if not combat.turn_end_started or (
        isinstance(effect_pending, EffectCardDiscoverSelect)
        and effect_pending.card_pile == CardPile.Draw
    ):
        # The first pass: Orichalcum, Plated Armor, the Cards that play themselves out of hand
        if walk.block == 0 and RelicName.Orichalcum in relic_names:
            walk.block = _ORICHALCUM_BLOCK
        walk.block += stacks_of_modifier_kind.get(ModifierKind.PlatedArmor, 0)
        hand = combat.card_pile_hand
        for card in hand:
            match card.name:
                case CardName.Burn:
                    damage = _BURN_UPGRADED_DAMAGE if card.upgraded else _BURN_DAMAGE
                    _deal_damage(walk, damage, from_monster=False)
                case CardName.Decay:
                    _deal_damage(walk, _DECAY_DAMAGE, from_monster=False)
                case CardName.Regret:
                    walk.damage += 1 if walk.intangible else len(hand)
                    _lose_health(walk, len(hand))

        # The landing: Regeneration heals ahead of its other hooks
        if RelicName.MarkOfTheBloom not in relic_names:
            regeneration = stacks_of_modifier_kind.get(ModifierKind.Regeneration, 0)
            walk.health = min(walk.health + regeneration, character.health_max)

    # Constricted's hit has no source
    constricted = stacks_of_modifier_kind.get(ModifierKind.Constricted, 0)
    _deal_damage(walk, constricted, from_monster=False)

    # The Monsters' turn: every tick, then the moves in slot order
    health_at_monsters = walk.health + sum(walk.revives)
    stacks_of_monster = {
        monster.id: {modifier.kind: modifier.stacks for modifier in monster.modifiers}
        for monster in monsters
    }
    health_of_monster = {}
    for monster in monsters:
        stacks = stacks_of_monster[monster.id]
        health = monster.health - get_poison_tick(
            stacks.get(ModifierKind.Poison, 0),
            ModifierKind.Intangible in stacks,
            stacks,
            game_state.ascension,
        )

        # Block expires at its turn start unless Barricade keeps it, and no tick touches it
        if health > 0 and ModifierKind.Barricade in stacks:
            health += monster.block
        health_of_monster[monster.id] = health
    exploded: set[int] = set()
    _resolve_deaths(monsters, stacks_of_monster, health_of_monster, exploded)
    thorns = stacks_of_modifier_kind.get(ModifierKind.Thorns, 0)
    damage_of_monster = {}
    health_lost_of_monster = {}
    for monster in monsters:
        damage_before, health_lost_before = walk.damage, walk.health_lost
        health_of_monster[monster.id] = _walk_monster_move(
            walk, monster, stacks_of_monster[monster.id], health_of_monster[monster.id], thorns
        )

        # A death in its move sets off what a tick's does
        if health_of_monster[monster.id] <= 0:
            _resolve_deaths(monsters, stacks_of_monster, health_of_monster, exploded)
        damage_of_monster[monster.id] = walk.damage - damage_before
        health_lost_of_monster[monster.id] = walk.health_lost - health_lost_before

    return Incoming(
        damage=walk.damage,
        health_lost=walk.health_lost,
        is_lethal=walk.is_lethal,
        health_at_monsters=health_at_monsters,
        damage_of_monster=damage_of_monster,
        health_lost_of_monster=health_lost_of_monster,
    )
