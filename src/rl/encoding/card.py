from typing import get_args

import numpy as np
from slai import MAX_ENERGY_MAX
from slai import MAX_MONSTERS
from slai import NUM_CHEMICAL_X_REPEATS
from slai import Card
from slai import CardColor
from slai import CardCostKind
from slai import CardCostKindXCost
from slai import CardKind
from slai import CardName
from slai import CardRarity
from slai import CostScope
from slai import EffectDamageFinisher
from slai import EffectDamageFlechettes
from slai import EffectDamageMindBlast
from slai import EffectDamagePhysical
from slai import EffectHeelHookProc
from slai import EffectSkewerProc
from slai import GameState
from slai import ModifierKind
from slai import PlaySource
from slai import RelicName
from slai import get_card_templates

from src.rl.constants import MAX_SIZE_CARD_PILE_DRAW
from src.rl.encoding.damage import deal_damage
from src.rl.encoding.damage import get_stacks_of_modifier_kind
from src.rl.encoding.damage import make_combatant
from src.rl.encoding.layout import make_channel_allocator
from src.rl.encoding.shared import encode_shared_outgoing_into
from src.rl.encoding.shared import encode_shared_price_into
from src.rl.encoding.shared import encode_shared_stasis_slot_into
from src.rl.encoding.shared import encode_shared_tag_into
from src.rl.encoding.shared import get_record_key_price
from src.rl.tokens import get_entities
from src.rl.tokens import get_monster_entities
from src.rl.tokens import relic_held
from src.rl.types import EntityKind
from src.rl.types import Outgoing
from src.rl.types import RecordKeyCard
from src.rl.types import Tag


# Enum maps, keyed by int(): the declaration-order discriminant
_IDX_OF_CARD_NAME = {card_name: int(card_name) for card_name in CardName.members()}
_IDX_OF_CARD_KIND = {card_kind: int(card_kind) for card_kind in CardKind.members()}
_IDX_OF_CARD_COLOR = {card_color: int(card_color) for card_color in CardColor.members()}
_IDX_OF_CARD_RARITY = {card_rarity: int(card_rarity) for card_rarity in CardRarity.members()}
_IDX_OF_COST_SCOPE = {scope: int(scope) for scope in CostScope.members()}
_IDX_OF_PLAY_SOURCE = {play_source: int(play_source) for play_source in PlaySource.members()}
_IDX_OF_CARD_COST_KIND = {cls: idx_class for idx_class, cls in enumerate(get_args(CardCostKind))}

# Cost, derived: the engine's largest max energy
_COST_MIN = 0
_COST_MAX = MAX_ENERGY_MAX
_COST_VECTOR_DIM = _COST_MAX - _COST_MIN + 1
_COST_OFFSET_MAX = max(
    1.0,
    max(
        (
            abs(card_template.cost_kind.offset)
            for card_template in get_card_templates()
            if isinstance(card_template.cost_kind, CardCostKindXCost)
        ),
        default=0,
    ),
)


# Per-card identity + energy + draw depth


_CHANNEL_LAYOUT_CARD, _alloc = make_channel_allocator()
_OFF_CARD_NAME = _alloc("name", len(_IDX_OF_CARD_NAME))
_OFF_CARD_KIND = _alloc("kind", len(_IDX_OF_CARD_KIND))
_OFF_CARD_COLOR = _alloc("color", len(_IDX_OF_CARD_COLOR))
_OFF_CARD_RARITY = _alloc("rarity", len(_IDX_OF_CARD_RARITY))
_OFF_CARD_COST_KIND = _alloc("cost_kind", len(_IDX_OF_CARD_COST_KIND))
_OFF_CARD_COST_SCOPE = _alloc("cost_scope", len(_IDX_OF_COST_SCOPE))
_OFF_CARD_COST_VECTOR = _alloc("cost_vector", _COST_VECTOR_DIM)
_OFF_CARD_COST_SCALAR = _alloc("cost_scalar")
_OFF_CARD_UPGRADED = _alloc("upgraded")
_OFF_CARD_EXHAUST = _alloc("exhaust")
_OFF_CARD_INNATE = _alloc("innate")
_OFF_CARD_ETHEREAL = _alloc("ethereal")
_OFF_CARD_RETAIN = _alloc("retain")
_OFF_CARD_REQUIRES_TARGET = _alloc("requires_target")
_OFF_CARD_PLAYABLE = _alloc("playable")
_OFF_CARD_ENERGY_DELTA = _alloc("energy_delta")
_OFF_CARD_COST_BASE = _alloc("cost_base")
_OFF_CARD_BOTTLED = _alloc("bottled")
_OFF_CARD_COST_OFFSET = _alloc("cost_offset")
_OFF_CARD_FREE_TO_PLAY_ONCE = _alloc("free_to_play_once")
_OFF_CARD_PLAY_SOURCE = _alloc("play_source", len(_IDX_OF_PLAY_SOURCE))
_OFF_CARD_DRAW_DEPTH = _alloc("draw_depth", MAX_SIZE_CARD_PILE_DRAW)
_OFF_CARD_DRAW_DEPTH_SCALAR = _alloc("draw_depth_scalar")
_OFF_CARD_DRAW_BELOW_SCALAR = _alloc("draw_below_scalar")

ENCODING_DIM_CARD = sum(w for _, w, _ in _CHANNEL_LAYOUT_CARD)

# Most nonzero columns a Card row holds: one per channel, each a one-hot or a scalar
ENCODING_DIM_CARD_SPARSE = sum(nonzeros for _, _, nonzeros in _CHANNEL_LAYOUT_CARD)

# Each tagged Card's Tags, by name: an upgraded Card keeps them
_TAGS_OF_CARD_NAME: dict[CardName, tuple[Tag, ...]] = {
    CardName.Accuracy: (Tag.NINJA,),
    CardName.Acrobatics: (Tag.DISCARD,),
    CardName.AfterImage: (Tag.NINJA,),
    CardName.AllOutAttack: (Tag.DISCARD,),
    CardName.AThousandCuts: (Tag.NINJA,),
    CardName.Backstab: (Tag.NINJA,),
    CardName.Bane: (Tag.POISON,),
    CardName.BladeDance: (Tag.NINJA,),
    CardName.BouncingFlask: (Tag.POISON,),
    CardName.CalculatedGamble: (Tag.DISCARD,),
    CardName.Catalyst: (Tag.POISON,),
    CardName.Choke: (Tag.NINJA,),
    CardName.CloakAndDagger: (Tag.NINJA,),
    CardName.Concentrate: (Tag.DISCARD,),
    CardName.CorpseExplosion: (Tag.POISON,),
    CardName.CripplingPoison: (Tag.POISON,),
    CardName.DaggerThrow: (Tag.DISCARD,),
    CardName.DeadlyPoison: (Tag.POISON,),
    CardName.EndlessAgony: (Tag.NINJA,),
    CardName.Envenom: (Tag.POISON, Tag.NINJA),
    CardName.Eviscerate: (Tag.DISCARD,),
    CardName.Finisher: (Tag.NINJA,),
    CardName.FlashOfSteel: (Tag.NINJA,),
    CardName.InfiniteBlades: (Tag.NINJA,),
    CardName.Neutralize: (Tag.NINJA,),
    CardName.NoxiousFumes: (Tag.POISON,),
    CardName.Panache: (Tag.NINJA,),
    CardName.PoisonedStab: (Tag.POISON,),
    CardName.Prepared: (Tag.DISCARD,),
    CardName.Reflex: (Tag.DISCARD,),
    CardName.Shiv: (Tag.NINJA,),
    CardName.SneakyStrike: (Tag.DISCARD,),
    CardName.StormOfSteel: (Tag.NINJA,),
    CardName.Survivor: (Tag.DISCARD,),
    CardName.Tactician: (Tag.DISCARD,),
    CardName.ToolsOfTheTrade: (Tag.DISCARD,),
    CardName.Unload: (Tag.DISCARD,),
}


def _cost_offset(card: Card) -> float:
    return float(card.cost_kind.offset if isinstance(card.cost_kind, CardCostKindXCost) else 0)


def encode_card_into(
    card: Card,
    energy: int,
    draw_below: int | None,
    gold: int,
    slot_holder: int | None,
    play_source: PlaySource | None,
    outgoing: Outgoing | None,
    np_own: np.ndarray,
    np_shared: np.ndarray,
) -> None:
    """Write a Card's token: its own columns, then its price, holder's slot, outgoing and Tags.

    Every input besides the Card is in the Card's record key (`get_record_key_card`), so a new
    input goes in the key too.

    Parameters
    ----------
    card : Card
        The Card.
    energy : int
        What its cost weighs against: the Character's energy in Combat, a turn's outside it.
    draw_below : int | None
        The Cards below it in the draw pile; None when its draw depth doesn't show.
    gold : int
        The Character's gold, which its price is read against.
    slot_holder : int | None
        The slot of the Monster holding it in stasis; None out of the stasis pile.
    play_source : PlaySource | None
        Its first CardPlay's source while it is queued; None out of the play queue.
    outgoing : Outgoing | None
        What playing it would take off each slot's Monster (`project_outgoing`); None out of the
        hand.
    np_own : np.ndarray
        (ENCODING_DIM_CARD,) float32. The Card's own range of its token row, written in place.
    np_shared : np.ndarray
        (ENCODING_DIM_SHARED,) float32. The token's shared block, written in place.
    """
    np_own[_OFF_CARD_NAME + _IDX_OF_CARD_NAME[card.name]] = 1.0
    np_own[_OFF_CARD_KIND + _IDX_OF_CARD_KIND[card.kind]] = 1.0
    np_own[_OFF_CARD_COLOR + _IDX_OF_CARD_COLOR[card.color]] = 1.0
    np_own[_OFF_CARD_RARITY + _IDX_OF_CARD_RARITY[card.rarity]] = 1.0
    np_own[_OFF_CARD_COST_KIND + _IDX_OF_CARD_COST_KIND[type(card.cost_kind)]] = 1.0
    if card.cost_override_scope is not None:
        np_own[_OFF_CARD_COST_SCOPE + _IDX_OF_COST_SCOPE[card.cost_override_scope]] = 1.0

    cost = max(_COST_MIN, min(card.cost, _COST_MAX))
    np_own[_OFF_CARD_COST_VECTOR + cost] = 1.0

    np_own[_OFF_CARD_COST_SCALAR] = cost / _COST_MAX
    np_own[_OFF_CARD_UPGRADED] = float(card.upgraded)
    np_own[_OFF_CARD_EXHAUST] = float(card.exhaust)
    np_own[_OFF_CARD_INNATE] = float(card.innate)
    np_own[_OFF_CARD_ETHEREAL] = float(card.ethereal)
    np_own[_OFF_CARD_RETAIN] = float(card.retain)
    np_own[_OFF_CARD_REQUIRES_TARGET] = float(card.requires_target)
    np_own[_OFF_CARD_PLAYABLE] = float(card.playable)
    np_own[_OFF_CARD_ENERGY_DELTA] = (
        max(-_COST_MAX, min(energy - card.cost, _COST_MAX)) / _COST_MAX
    )
    np_own[_OFF_CARD_COST_BASE] = max(_COST_MIN, min(card.cost_base, _COST_MAX)) / _COST_MAX
    np_own[_OFF_CARD_BOTTLED] = float(card.bottled)
    np_own[_OFF_CARD_COST_OFFSET] = _cost_offset(card) / _COST_OFFSET_MAX
    np_own[_OFF_CARD_FREE_TO_PLAY_ONCE] = float(card.free_to_play_once)
    if play_source is not None:
        np_own[_OFF_CARD_PLAY_SOURCE + _IDX_OF_PLAY_SOURCE[play_source]] = 1.0
    if card.draw_depth is not None:
        np_own[_OFF_CARD_DRAW_DEPTH + min(card.draw_depth, MAX_SIZE_CARD_PILE_DRAW - 1)] = 1.0
        np_own[_OFF_CARD_DRAW_DEPTH_SCALAR] = (
            min(card.draw_depth, MAX_SIZE_CARD_PILE_DRAW) / MAX_SIZE_CARD_PILE_DRAW
        )
    if draw_below is not None:
        np_own[_OFF_CARD_DRAW_BELOW_SCALAR] = (
            min(draw_below, MAX_SIZE_CARD_PILE_DRAW) / MAX_SIZE_CARD_PILE_DRAW
        )

    # Shop price
    encode_shared_price_into(card.price, gold, np_shared)

    # Stasis slot. Shared w/ Monsters
    if slot_holder is not None:
        encode_shared_stasis_slot_into(slot_holder, np_shared)

    # Outgoing. Shared w/ Monsters
    if outgoing is not None:
        for slot, outgoing_slot in enumerate(outgoing):
            if outgoing_slot is not None:
                health_lost, is_lethal = outgoing_slot
                encode_shared_outgoing_into(slot, health_lost, is_lethal, np_shared)

    # Tags. Shared w/ Relics and Potions
    encode_shared_tag_into(_TAGS_OF_CARD_NAME.get(card.name, ()), np_shared)


def get_record_key_card(
    card: Card,
    energy: int,
    draw_below: int | None,
    gold: int,
    slot_holder: int | None,
    play_source: PlaySource | None,
    outgoing: Outgoing | None,
) -> RecordKeyCard:
    """Make a Card's record key: its identity_hash and everything else its row reads.

    Parameters
    ----------
    card : Card
        The Card.
    energy : int
        What its cost weighs against: the Character's energy in Combat, a turn's outside it.
    draw_below : int | None
        The Cards below it in the draw pile; None when its draw depth doesn't show.
    gold : int
        The Character's gold, which its price is read against.
    slot_holder : int | None
        The slot of the Monster holding it in stasis; None out of the stasis pile.
    play_source : PlaySource | None
        Its first CardPlay's source while it is queued; None out of the play queue.
    outgoing : Outgoing | None
        What playing it would take off each slot's Monster (`project_outgoing`); None out of the
        hand.

    Returns
    -------
    RecordKeyCard
        Its identity_hash, the energy, the Cards below it, the gold when it has a price, its
        holder's slot while it is in stasis, its play source while it is queued, and what playing
        it would take off each slot's Monster while it is in the hand.
    """
    return (
        card.identity_hash,
        energy,
        draw_below,
        get_record_key_price(card.price, gold),
        slot_holder,
        play_source,
        outgoing,
    )


def project_outgoing(game_state: GameState) -> dict[int, Outgoing]:
    """Return what playing each hand Card now would take off each slot's Monster.

    Each hit is the damage the game shows with the Card hovered over that Monster
    (`Combat.damage_card_pile_hand`). The hits are one play's own, Bane's poisoned hit counted
    only when that Monster already holds Poison. Block takes the first hits, The Boot lifts a
    small remainder past it, and Invincible caps the HP lost at what it has left this turn. An
    area attack's values hold for every slot at once, a targeted Card's for aiming it at each
    slot.

    Parameters
    ----------
    game_state : GameState
        The state to read.

    Returns
    -------
    dict[int, Outgoing]
        Each hand Card's (HP lost, lethal) against each slot, by the Card's id; None at the slots
        where the engine shows it no hit. Empty outside Combat.
    """
    combat = game_state.combat
    if combat is None:
        return {}

    hand = get_entities(game_state, combat.id_card_pile_hand, Card)
    monsters = get_monster_entities(game_state)
    stacks_of_monster = {
        monster.id: get_stacks_of_modifier_kind(monster.modifiers) for monster in monsters
    }
    has_boot = relic_held(game_state, RelicName.Boot)

    # What Finisher, Flechettes and an X-cost Card count, as they stand before the play
    this_turn_attacks = combat.this_turn_attacks
    num_skills_in_hand = sum(card.kind == CardKind.Skill for card in hand)
    energy = combat.energy.energy_current
    has_chemical_x = relic_held(game_state, RelicName.ChemicalX)

    outgoing_of_card: dict[int, Outgoing] = {}
    for card, damage_card in zip(hand, combat.damage_card_pile_hand):
        # Every Monster takes the first hits, only a poisoned one Bane's second
        hits = hits_poisoned = 0
        for effect in card.effects_play:
            match effect:
                case EffectDamagePhysical(if_poisoned=True):
                    hits_poisoned += effect.instances
                case EffectDamagePhysical():
                    hits += effect.instances
                case EffectDamageFinisher():
                    hits += this_turn_attacks
                case EffectDamageFlechettes():
                    hits += num_skills_in_hand
                case EffectHeelHookProc() | EffectSkewerProc() | EffectDamageMindBlast():
                    hits += 1

        # An X-cost Card plays its Effects X times
        num_repeats = 1
        if isinstance(card.cost_kind, CardCostKindXCost):
            num_repeats = max(energy + card.cost_kind.offset, 0)
            if has_chemical_x:
                num_repeats += NUM_CHEMICAL_X_REPEATS

        outgoing: list[tuple[int, bool] | None] = [None] * MAX_MONSTERS
        for monster in monsters:
            damage = damage_card[monster.slot]
            if damage is None:
                continue

            stacks_of_modifier_kind_monster = stacks_of_monster[monster.id]
            hits_monster = hits
            if ModifierKind.Poison in stacks_of_modifier_kind_monster:
                hits_monster += hits_poisoned

            # The hits land one by one on a copy of its block and HP
            combatant_monster = make_combatant(
                EntityKind.MONSTER,
                monster.health,
                monster.block,
                stacks_of_modifier_kind_monster,
                has_boot=has_boot,
            )
            for _ in range(hits_monster * num_repeats):
                deal_damage(combatant_monster, damage, EntityKind.CARD)
            outgoing[monster.slot] = (combatant_monster.health_lost, combatant_monster.is_lethal)
        outgoing_of_card[card.id] = tuple(outgoing)

    return outgoing_of_card
