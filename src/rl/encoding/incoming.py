"""What ending the Character's turn now would cost it.

Walks what the engine runs from the turn end to the last Monster's move, on the state as it stands:
block gained at the turn end, Cards that play themselves, Poison ticks and the deaths they set off
ahead of the moves, and each hit through the rules that change what it costs.
"""

from dataclasses import dataclass
from typing import cast

from slai import BURN_DAMAGE
from slai import BURN_DAMAGE_UPGRADED
from slai import DECAY_DAMAGE
from slai import EXPLOSIVE_DAMAGE
from slai import ORICHALCUM_BLOCK
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

from src.rl.encoding.damage import Combatant
from src.rl.encoding.damage import deal_damage
from src.rl.encoding.damage import get_poison_tick
from src.rl.encoding.damage import get_stacks_of_modifier_kind
from src.rl.encoding.damage import lose_health
from src.rl.encoding.damage import make_combatant
from src.rl.tokens import get_character
from src.rl.tokens import get_entities
from src.rl.tokens import get_monster_entities
from src.rl.types import EntityKind


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


def _resolve_deaths(
    monsters: list[Monster],
    stacks_of_monster: dict[int, dict[ModifierKind, int]],
    combatant_of_monster: dict[int, Combatant],
    id_monsters_exploded: set[int],
) -> None:
    """Resolve what the Monsters' deaths set off, ahead of the next move.

    Parameters
    ----------
    monsters : list[Monster]
        The Monsters.
    stacks_of_monster : dict[int, dict[ModifierKind, int]]
        Each Monster's Modifiers, by id: each kind's stacks.
    combatant_of_monster : dict[int, Combatant]
        Each Monster's side of the walk by id; 0 HP or less for one that won't move. Moved along
        in place.
    id_monsters_exploded : set[int]
        The ids whose Corpse Explosion went off. Moved along in place.
    """
    # Corpse Explosion: a death deals its max health per stack to the others, which can chain
    for _ in monsters:
        for monster in monsters:
            corpse_explosion = stacks_of_monster[monster.id].get(ModifierKind.CorpseExplosion, 0)
            if (
                corpse_explosion > 0
                and combatant_of_monster[monster.id].health <= 0
                and not monster.half_dead
                and monster.id not in id_monsters_exploded
            ):
                id_monsters_exploded.add(monster.id)
                for other in monsters:
                    combatant_other = combatant_of_monster[other.id]

                    # Corpses and half-dead Monsters, at 0 HP or less, absorb nothing
                    if combatant_other.health > 0:
                        deal_damage(combatant_other, monster.health_max * corpse_explosion, None)

    # The Awakened One's final death sends its Cultists running
    for monster in monsters:
        if (
            monster.name == MonsterName.AwakenedOne
            and not monster.half_dead
            and ModifierKind.Unawakened not in stacks_of_monster[monster.id]
            and combatant_of_monster[monster.id].health <= 0
        ):
            for other in monsters:
                if other.name == MonsterName.Cultist:
                    combatant_of_monster[other.id].health = 0

    # A leader's death takes its Minions along, or sends them running
    monsters_dead = [
        monster for monster in monsters if combatant_of_monster[monster.id].health <= 0
    ]
    monsters_alive = [
        monster for monster in monsters if combatant_of_monster[monster.id].health > 0
    ]
    if any(
        ModifierKind.Minion not in stacks_of_monster[monster.id] for monster in monsters_dead
    ) and all(ModifierKind.Minion in stacks_of_monster[monster.id] for monster in monsters_alive):
        for monster in monsters_alive:
            combatant_of_monster[monster.id].health = 0


def _walk_monster_move(
    combatant_char: Combatant,
    monster: Monster,
    stacks_of_modifier_kind_monster: dict[ModifierKind, int],
    combatant_monster: Combatant,
    thorns_char: int,
) -> None:
    """Walk the Monster's move.

    Parameters
    ----------
    combatant_char : Combatant
        The Character's side of the walk, moved along in place.
    monster : Monster
        The Monster, with its intent visible.
    stacks_of_modifier_kind_monster : dict[ModifierKind, int]
        The Monster's Modifiers: each kind's stacks.
    combatant_monster : Combatant
        Its side of the walk at its move, moved along in place; 0 HP or less once it's dead.
    thorns_char : int
        The Character's Thorns stacks.
    """
    # A dead Monster skips its move, and a half-dead one hits nothing
    if monster.half_dead or combatant_monster.health <= 0:
        return

    # HP lost ahead of the move that crosses Split or Mode Shift replaces the move
    health_lost_monster = combatant_monster.health_lost
    is_splittable_monster = ModifierKind.Splittable in stacks_of_modifier_kind_monster
    mode_shift_monster = stacks_of_modifier_kind_monster.get(ModifierKind.ModeShift, 0)
    if health_lost_monster > 0 and (
        (is_splittable_monster and combatant_monster.health <= monster.health_max // 2)
        or 0 < mode_shift_monster <= health_lost_monster
    ):
        return

    # Shifting: HP lost comes off its Strength until its turn end
    intent = cast(Intent, monster.intent)
    hit = intent.damage or 0
    if ModifierKind.Shifting in stacks_of_modifier_kind_monster:
        hit = max(hit - health_lost_monster, 0)

    # Thorns can kill it mid-move, which cancels its remaining hits
    for _ in range(intent.instances or 0):
        if combatant_monster.health <= 0:
            break

        deal_damage(combatant_char, hit, EntityKind.MONSTER)
        deal_damage(combatant_monster, thorns_char, None)

    # Explosive's last stack blows up behind the move, with no source, and takes the Monster along
    if stacks_of_modifier_kind_monster.get(ModifierKind.Explosive, 0) == 1:
        deal_damage(combatant_char, EXPLOSIVE_DAMAGE, None)
        combatant_monster.health = 0


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
    stacks_of_modifier_kind_char = get_stacks_of_modifier_kind(character.modifiers)
    relic_names = {relic.name for relic in get_entities(game_state, game_state.id_relics, Relic)}

    # Initialize walk
    combatant_char = make_combatant(
        EntityKind.CHARACTER,
        character.health,
        character.block,
        stacks_of_modifier_kind_char,
        has_torii=RelicName.Torii in relic_names,
        has_tungsten_rod=RelicName.TungstenRod in relic_names,
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
        if combatant_char.block == 0 and RelicName.Orichalcum in relic_names:
            combatant_char.block = ORICHALCUM_BLOCK
        combatant_char.block += stacks_of_modifier_kind_char.get(ModifierKind.PlatedArmor, 0)
        hand = get_entities(game_state, combat.id_card_pile_hand, Card)
        for card in hand:
            match card.name:
                case CardName.Burn:
                    damage = BURN_DAMAGE_UPGRADED if card.upgraded else BURN_DAMAGE
                    deal_damage(combatant_char, damage, None)
                case CardName.Decay:
                    deal_damage(combatant_char, DECAY_DAMAGE, None)
                case CardName.Regret:
                    combatant_char.damage += lose_health(combatant_char, len(hand), None)

        # The landing: Regeneration heals ahead of its other hooks
        if RelicName.MarkOfTheBloom not in relic_names:
            regeneration_char = stacks_of_modifier_kind_char.get(ModifierKind.Regeneration, 0)
            combatant_char.health = min(
                combatant_char.health + regeneration_char, character.health_max
            )

    # Constricted's hit has no source
    constricted_char = stacks_of_modifier_kind_char.get(ModifierKind.Constricted, 0)
    deal_damage(combatant_char, constricted_char, None)

    # The Monsters' turn: every tick, then the moves in slot order
    health_at_monsters = combatant_char.health + sum(combatant_char.revives)
    stacks_of_monster = {
        monster.id: get_stacks_of_modifier_kind(monster.modifiers) for monster in monsters
    }
    combatant_of_monster = {}
    for monster in monsters:
        stacks_of_modifier_kind_monster = stacks_of_monster[monster.id]

        # Block expires at its turn start unless Barricade keeps it, and no tick touches it
        is_barricade_monster = ModifierKind.Barricade in stacks_of_modifier_kind_monster
        combatant_monster = make_combatant(
            EntityKind.MONSTER,
            monster.health,
            monster.block if is_barricade_monster else 0,
            stacks_of_modifier_kind_monster,
        )
        get_poison_tick(
            combatant_monster,
            stacks_of_modifier_kind_monster.get(ModifierKind.Poison, 0),
            game_state.ascension,
        )
        combatant_of_monster[monster.id] = combatant_monster
    id_monsters_exploded: set[int] = set()
    _resolve_deaths(monsters, stacks_of_monster, combatant_of_monster, id_monsters_exploded)
    thorns_char = stacks_of_modifier_kind_char.get(ModifierKind.Thorns, 0)
    damage_of_monster = {}
    health_lost_of_monster = {}
    for monster in monsters:
        damage_before, health_lost_before = combatant_char.damage, combatant_char.health_lost
        _walk_monster_move(
            combatant_char,
            monster,
            stacks_of_monster[monster.id],
            combatant_of_monster[monster.id],
            thorns_char,
        )

        # A death in its move sets off what a tick's does
        if combatant_of_monster[monster.id].health <= 0:
            _resolve_deaths(
                monsters, stacks_of_monster, combatant_of_monster, id_monsters_exploded
            )
        damage_of_monster[monster.id] = combatant_char.damage - damage_before
        health_lost_of_monster[monster.id] = combatant_char.health_lost - health_lost_before

    return Incoming(
        damage=combatant_char.damage,
        health_lost=combatant_char.health_lost,
        is_lethal=combatant_char.is_lethal,
        health_at_monsters=health_at_monsters,
        damage_of_monster=damage_of_monster,
        health_lost_of_monster=health_lost_of_monster,
    )
