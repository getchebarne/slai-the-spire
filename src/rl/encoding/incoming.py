"""What ending the Character's turn now would cost it.

Walks what the engine runs from the turn end to the last Monster's move, on the state as it stands:
block gained at the turn end, Cards that play themselves, Poison ticks and the deaths they set off
ahead of the moves, and each hit through the rules that change what it costs.
"""

from dataclasses import dataclass
from typing import cast

from slai import AmountRelative
from slai import Card
from slai import CardName
from slai import CardPile
from slai import EffectCardDiscoverSelect
from slai import EffectHealthDelta
from slai import GameState
from slai import Intent
from slai import ModifierKind
from slai import Monster
from slai import MonsterName
from slai import Potion
from slai import PotionName
from slai import Relic
from slai import RelicName

from src.rl.constants import INVINCIBLE_STACKS_OF_ASCENSION
from src.rl.tokens import get_character
from src.rl.tokens import get_entities
from src.rl.tokens import get_monster_entities


# The engine's hardcoded amounts, which GameEnv doesn't expose TODO: expose
_ORICHALCUM_BLOCK = 6
_BURN_DAMAGE = 2
_BURN_DAMAGE_UPGRADED = 4
_DECAY_DAMAGE = 2
_EXPLOSIVE_DAMAGE = 30

# A Monster hit leaving this much past block lands 1 under Torii
_TORII_MIN = 2
_TORII_MAX = 5


@dataclass
class _Walk:
    """The Character's side of the walk, moved along by each hit."""

    health_char: int
    block_char: int
    buffer_instances_char: int  # HP-loss instances Buffer still absorbs
    is_intangible_char: bool
    has_torii: bool
    has_tungsten_rod: bool
    revives_char: list[int]  # The HP each revive left restores, the next one first
    damage: int = 0          # Dealt so far: what landed on block and HP
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
    poison_monster: int,
    is_intangible_monster: bool,
    stacks_of_modifier_kind_monster: dict[ModifierKind, int],
    ascension: int,
) -> int:
    """Return the HP a tick of `poison_monster` takes from a Monster holding those Modifiers.

    Parameters
    ----------
    poison_monster : int
        The Poison stacks.
    is_intangible_monster : bool
        Whether Intangible covers the tick.
    stacks_of_modifier_kind_monster : dict[ModifierKind, int]
        The Monster's Modifiers: each kind's stacks.
    ascension : int
        The run's Ascension, which sets Invincible's cap.

    Returns
    -------
    int
        The HP the tick takes.
    """
    if is_intangible_monster:
        return min(poison_monster, 1)

    # Invincible's cap refills ahead of its holder's tick
    if ModifierKind.Invincible in stacks_of_modifier_kind_monster:
        return min(poison_monster, INVINCIBLE_STACKS_OF_ASCENSION[ascension])

    return poison_monster


def _lose_health(walk: _Walk, amount: int) -> None:
    # Intangible clamps the instance
    if walk.is_intangible_char:
        amount = min(amount, 1)

    # Buffer absorbs it whole
    if walk.buffer_instances_char > 0:
        walk.buffer_instances_char -= 1

        return

    # Tungsten Rod takes 1 off
    if walk.has_tungsten_rod:
        amount -= 1

    # Accumulate lost health
    walk.health_char -= amount
    walk.health_lost += amount

    # Lethal check
    if walk.health_char <= 0:
        if walk.revives_char:
            walk.health_char = walk.revives_char.pop(0)
        else:
            walk.is_lethal = True


def _deal_damage(walk: _Walk, amount: int, is_from_monster: bool) -> None:
    if walk.is_intangible_char:
        amount = min(amount, 1)

    # Accumulate damage taken
    walk.damage += amount

    # Calculate damage over block, set new current block
    damage_over_block = max(amount - walk.block_char, 0)
    walk.block_char = max(walk.block_char - amount, 0)

    # Apply Torii reduction
    if is_from_monster and walk.has_torii and _TORII_MIN <= damage_over_block <= _TORII_MAX:
        damage_over_block = 1

    # Accumulate lost health if any
    if damage_over_block > 0:
        _lose_health(walk, damage_over_block)


def _resolve_deaths(
    monsters: list[Monster],
    stacks_of_monster: dict[int, dict[ModifierKind, int]],
    health_of_monster: dict[int, int],
    id_monsters_exploded: set[int],
) -> None:
    """Resolve what the Monsters' deaths set off, ahead of the next move.

    Parameters
    ----------
    monsters : list[Monster]
        The Monsters.
    stacks_of_monster : dict[int, dict[ModifierKind, int]]
        Each Monster's Modifiers, by id: each kind's stacks.
    health_of_monster : dict[int, int]
        Each Monster's health by id, with the block Barricade keeps; 0 or less for one that won't
        move. Moved along in place.
    id_monsters_exploded : set[int]
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
                and monster.id not in id_monsters_exploded
            ):
                id_monsters_exploded.add(monster.id)
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
    monsters_dead = [monster for monster in monsters if health_of_monster[monster.id] <= 0]
    monsters_alive = [monster for monster in monsters if health_of_monster[monster.id] > 0]
    if any(
        ModifierKind.Minion not in stacks_of_monster[monster.id] for monster in monsters_dead
    ) and all(ModifierKind.Minion in stacks_of_monster[monster.id] for monster in monsters_alive):
        for monster in monsters_alive:
            health_of_monster[monster.id] = 0


def _walk_monster_move(
    walk: _Walk,
    monster: Monster,
    stacks_of_modifier_kind_monster: dict[ModifierKind, int],
    health_monster: int,
    thorns_char: int,
) -> int:
    """Walk the Monster's move.

    Parameters
    ----------
    walk : _Walk
        The walk, moved along in place.
    monster : Monster
        The Monster, with its intent visible.
    stacks_of_modifier_kind_monster : dict[ModifierKind, int]
        The Monster's Modifiers: each kind's stacks.
    health_monster : int
        Its health at its move, with the block Barricade keeps.
    thorns_char : int
        The Character's Thorns stacks.

    Returns
    -------
    int
        Its health after the move, likewise; 0 or less for one that's dead.
    """
    # A dead Monster skips its move, and a half-dead one hits nothing
    if monster.half_dead or health_monster <= 0:
        return health_monster

    # HP lost ahead of the move that crosses Split or Mode Shift replaces the move
    health_lost_monster = monster.health - health_monster
    is_splittable_monster = ModifierKind.Splittable in stacks_of_modifier_kind_monster
    mode_shift_monster = stacks_of_modifier_kind_monster.get(ModifierKind.ModeShift, 0)
    if health_lost_monster > 0 and (
        (is_splittable_monster and health_monster <= monster.health_max // 2)
        or 0 < mode_shift_monster <= health_lost_monster
    ):
        return health_monster

    # Shifting: HP lost comes off its Strength until its turn end
    intent = cast(Intent, monster.intent)
    hit = intent.damage or 0
    if ModifierKind.Shifting in stacks_of_modifier_kind_monster:
        hit = max(hit - health_lost_monster, 0)

    # Thorns can kill it mid-move, which cancels its remaining hits
    if ModifierKind.Intangible in stacks_of_modifier_kind_monster:
        thorns_char = min(thorns_char, 1)

    for _ in range(intent.instances or 0):
        if health_monster <= 0:
            break

        _deal_damage(walk, hit, is_from_monster=True)
        health_monster -= thorns_char

    # Explosive's last stack blows up behind the move, with no source, and takes the Monster along
    if stacks_of_modifier_kind_monster.get(ModifierKind.Explosive, 0) == 1:
        _deal_damage(walk, _EXPLOSIVE_DAMAGE, is_from_monster=False)

        return 0

    return health_monster


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
    health_max_char = get_character(game_state).health_max
    revives_char = []
    for potion in get_entities(game_state, game_state.id_potions, Potion):
        if potion.name == PotionName.Fairy:
            amount = cast(AmountRelative, cast(EffectHealthDelta, potion.effects[0]).amount)
            revives_char.append(max(health_max_char * amount.numerator // amount.denominator, 1))

    relics = get_entities(game_state, game_state.id_relics, Relic)
    if any(relic.name == RelicName.LizardTail and not relic.used_up for relic in relics):
        revives_char.append(max(health_max_char // 2, 1))

    return revives_char


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

    monsters = get_monster_entities(game_state)
    if any(monster.intent is None for monster in monsters):
        return None

    character = get_character(game_state)
    stacks_of_modifier_kind_char = {
        modifier.kind: modifier.stacks for modifier in character.modifiers
    }
    relic_names = {relic.name for relic in get_entities(game_state, game_state.id_relics, Relic)}

    # Initialize walk
    walk = _Walk(
        health_char=character.health,
        block_char=character.block,
        buffer_instances_char=stacks_of_modifier_kind_char.get(ModifierKind.Buffer, 0),
        is_intangible_char=ModifierKind.Intangible in stacks_of_modifier_kind_char,
        has_torii=RelicName.Torii in relic_names,
        has_tungsten_rod=RelicName.TungstenRod in relic_names,
        revives_char=[] if RelicName.MarkOfTheBloom in relic_names else _get_revives(game_state),
    )

    # A selection inside the turn end comes after its first pass and the heal, but for Nilry's
    # Codex in the first pass, with Orichalcum already landed
    effect_pending = game_state.effect_pending
    if not combat.turn_end_started or (
        isinstance(effect_pending, EffectCardDiscoverSelect)
        and effect_pending.card_pile == CardPile.Draw
    ):
        # The first pass: Orichalcum, Plated Armor, the Cards that play themselves out of hand
        if walk.block_char == 0 and RelicName.Orichalcum in relic_names:
            walk.block_char = _ORICHALCUM_BLOCK
        walk.block_char += stacks_of_modifier_kind_char.get(ModifierKind.PlatedArmor, 0)
        hand = get_entities(game_state, combat.id_card_pile_hand, Card)
        for card in hand:
            match card.name:
                case CardName.Burn:
                    damage = _BURN_DAMAGE_UPGRADED if card.upgraded else _BURN_DAMAGE
                    _deal_damage(walk, damage, is_from_monster=False)
                case CardName.Decay:
                    _deal_damage(walk, _DECAY_DAMAGE, is_from_monster=False)
                case CardName.Regret:
                    walk.damage += 1 if walk.is_intangible_char else len(hand)
                    _lose_health(walk, len(hand))

        # The landing: Regeneration heals ahead of its other hooks
        if RelicName.MarkOfTheBloom not in relic_names:
            regeneration_char = stacks_of_modifier_kind_char.get(ModifierKind.Regeneration, 0)
            walk.health_char = min(walk.health_char + regeneration_char, character.health_max)

    # Constricted's hit has no source
    constricted_char = stacks_of_modifier_kind_char.get(ModifierKind.Constricted, 0)
    _deal_damage(walk, constricted_char, is_from_monster=False)

    # The Monsters' turn: every tick, then the moves in slot order
    health_at_monsters = walk.health_char + sum(walk.revives_char)
    stacks_of_monster = {
        monster.id: {modifier.kind: modifier.stacks for modifier in monster.modifiers}
        for monster in monsters
    }
    health_of_monster = {}
    for monster in monsters:
        stacks_of_modifier_kind_monster = stacks_of_monster[monster.id]
        health_monster = monster.health - get_poison_tick(
            stacks_of_modifier_kind_monster.get(ModifierKind.Poison, 0),
            ModifierKind.Intangible in stacks_of_modifier_kind_monster,
            stacks_of_modifier_kind_monster,
            game_state.ascension,
        )

        # Block expires at its turn start unless Barricade keeps it, and no tick touches it
        if health_monster > 0 and ModifierKind.Barricade in stacks_of_modifier_kind_monster:
            health_monster += monster.block
        health_of_monster[monster.id] = health_monster
    id_monsters_exploded: set[int] = set()
    _resolve_deaths(monsters, stacks_of_monster, health_of_monster, id_monsters_exploded)
    thorns_char = stacks_of_modifier_kind_char.get(ModifierKind.Thorns, 0)
    damage_of_monster = {}
    health_lost_of_monster = {}
    for monster in monsters:
        damage_before, health_lost_before = walk.damage, walk.health_lost
        health_of_monster[monster.id] = _walk_monster_move(
            walk,
            monster,
            stacks_of_monster[monster.id],
            health_of_monster[monster.id],
            thorns_char,
        )

        # A death in its move sets off what a tick's does
        if health_of_monster[monster.id] <= 0:
            _resolve_deaths(monsters, stacks_of_monster, health_of_monster, id_monsters_exploded)
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
