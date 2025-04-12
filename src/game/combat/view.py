# TODO: remove effect, entity_id from monster and card, complement w/ FSM in CombatState

from dataclasses import dataclass

from src.game.combat.effect import Effect
from src.game.combat.effect import EffectType
from src.game.combat.entities import Actor
from src.game.combat.entities import Card
from src.game.combat.entities import ModifierType
from src.game.combat.entities import Monster
from src.game.combat.entities import MonsterMove
from src.game.combat.state import CombatState
from src.game.combat.state import FSMState
from src.game.combat.utils import does_card_require_target


# Aliases
ModifierViewType = ModifierType
EffectView = Effect
FSMStateView = FSMState


@dataclass
class CardView:
    name: str
    effects: list[Effect]  # TODO: implement EffectView
    cost: int
    is_active: bool
    requires_target: bool
    entity_id: int


@dataclass
class ActorView:
    name: str
    health_current: int
    health_max: int
    block_current: int

    # Dictionary mapping modifier types to current stacks
    modifiers: dict[ModifierViewType, int | None]


@dataclass
class CharacterView(ActorView):
    pass


@dataclass
class IntentView:
    damage: int | None
    instances: int | None
    block: bool
    buff: bool


@dataclass
class MonsterView(ActorView):
    intent: IntentView
    entity_id: int


@dataclass
class EnergyView:
    current: int
    max: int


@dataclass
class CombatView:
    character: CharacterView
    monsters: list[MonsterView]
    hand: list[CardView]
    draw_pile: list[CardView]
    disc_pile: list[CardView]
    energy: EnergyView
    effect: EffectView | None
    state: FSMStateView


def get_card_view(card: Card, is_active: bool, id_card: int) -> CardView:
    requires_target = does_card_require_target(card)

    return CardView(card.name, card.effects, card.cost, is_active, requires_target, id_card)


def get_actor_view(actor: Actor) -> ActorView:
    return ActorView(
        actor.name,
        actor.health_current,
        actor.health_max,
        actor.block_current,
        {
            modifier_type: modifier.stacks_current
            for modifier_type, modifier in actor.modifiers.items()
        },
    )


def _move_to_intent(move: MonsterMove) -> IntentView:
    # Initialze empty intent
    intent = IntentView(None, None, False, False)

    # Iterate over the move's effects
    for effect in move.effects:
        if effect.type == EffectType.DEAL_DAMAGE:
            if intent.damage is None and intent.instances is None:
                intent.damage = effect.value
                intent.instances = 1

                continue

            if intent.damage != effect.value:
                raise ValueError("All of the move's damage instances must have the same value")

            intent.instances += 1

        if not intent.block and effect.type == EffectType.GAIN_BLOCK:
            intent.block = True

        # TODO: add support for other buffs
        if not intent.buff and effect.type == EffectType.GAIN_STRENGTH:
            intent.buff = True

    return intent


def _correct_intent_damage(damage: int, monster_view: MonsterView) -> int:
    if ModifierViewType.STRENGTH in monster_view.modifiers:
        damage += monster_view.modifiers[ModifierViewType.STRENGTH].stacks_current

    if ModifierType.WEAK in monster_view.modifiers:
        damage *= 0.75

    return int(damage)


def get_monster_view(monster: Monster, id_monster: int) -> MonsterView:
    actor_view = get_actor_view(monster)
    intent_view = _move_to_intent(monster.move_current)

    if intent_view.damage is not None:
        intent_view.damage = _correct_intent_damage(intent_view.damage, monster)

    return MonsterView(
        actor_view.name,
        actor_view.health_current,
        actor_view.health_max,
        actor_view.block_current,
        actor_view.modifiers,
        intent_view,
        id_monster,
    )


def view_combat(cs: CombatState) -> CombatView:
    # Character
    character = cs.entity_manager.entities[cs.entity_manager.id_character]
    character_view = get_actor_view(character)

    # Monsters
    monster_views = []
    for id_monster in cs.entity_manager.id_monsters:
        monster = cs.entity_manager.entities[id_monster]
        monster_views.append(get_monster_view(monster, id_monster))

    # Hand
    card_in_hand_views = []
    for id_card in cs.entity_manager.id_cards_in_hand:
        card = cs.entity_manager.entities[id_card]
        is_active = id_card == cs.entity_manager.id_card_active
        card_in_hand_views.append(get_card_view(card, is_active, id_card))

    # Draw pile
    card_in_draw_pile_views = []
    for id_card in cs.entity_manager.id_cards_in_draw_pile:
        card = cs.entity_manager.entities[id_card]

        if id_card == cs.entity_manager.id_card_active:
            raise ValueError(f"Card {card} with id {id_card} is in the draw pile but it's active")

        card_in_draw_pile_views.append(get_card_view(card, False, id_card))

    # Draw pile
    card_in_disc_pile_views = []
    for id_card in cs.entity_manager.id_cards_in_disc_pile:
        card = cs.entity_manager.entities[id_card]

        if id_card == cs.entity_manager.id_card_active:
            raise ValueError(
                f"Card {card} with id {id_card} is in the discard pile but it's active"
            )

        card_in_disc_pile_views.append(get_card_view(card, False, id_card))

    # Energy
    energy = cs.entity_manager.entities[cs.entity_manager.id_energy]
    energy_view = EnergyView(energy.current, energy.max)

    # Effect
    effect_view = None
    if cs.effect_queue:
        effect_view = cs.effect_queue[0]

    # FSM state
    state_view = cs.fsm_state

    return CombatView(
        character_view,
        monster_views,
        card_in_hand_views,
        card_in_draw_pile_views,
        card_in_disc_pile_views,
        energy_view,
        effect_view,
        state_view,
    )
