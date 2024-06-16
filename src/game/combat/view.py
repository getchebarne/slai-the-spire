from dataclasses import dataclass
from typing import Optional

from src.game.ecs.components.actors import ActorHasModifiersComponent
from src.game.ecs.components.actors import BlockComponent
from src.game.ecs.components.actors import CharacterComponent
from src.game.ecs.components.actors import HealthComponent
from src.game.ecs.components.actors import ModifierStacksComponent
from src.game.ecs.components.actors import MonsterComponent
from src.game.ecs.components.actors import MonsterCurrentMoveComponent
from src.game.ecs.components.actors import MonsterHasMovesComponent
from src.game.ecs.components.actors import MonsterMoveIntentComponent
from src.game.ecs.components.cards import CardCostComponent
from src.game.ecs.components.cards import CardInDiscardPileComponent
from src.game.ecs.components.cards import CardInDrawPileComponent
from src.game.ecs.components.cards import CardInHandComponent
from src.game.ecs.components.common import CanBeSelectedComponent
from src.game.ecs.components.common import DescriptionComponent
from src.game.ecs.components.common import IsSelectedComponent
from src.game.ecs.components.common import NameComponent
from src.game.ecs.components.effects import EffectIsPendingInputTargetsComponent
from src.game.ecs.components.effects import EffectNumberOfTargetsComponent
from src.game.ecs.components.energy import EnergyComponent
from src.game.ecs.manager import ECSManager


@dataclass
class Card:
    entity_id: int
    name: str
    description: str
    cost: int
    is_selected: bool
    can_be_selected: bool


@dataclass
class Energy:
    entity_id: int
    max: int
    current: int


@dataclass
class Health:
    max: int
    current: int


@dataclass
class Block:
    max: int
    current: int


@dataclass
class Modifier:
    entity_id: int
    name: str
    stacks: Optional[int]


@dataclass
class Actor:
    entity_id: int
    name: str
    health: Health
    block: Block
    modifiers: list[Modifier]


@dataclass
class Intent:
    damage: int
    times: int
    block: bool


# TODO: add intent
@dataclass
class Monster(Actor):
    can_be_selected: bool
    intent: Intent


@dataclass
class Character(Actor):
    pass


# TODO: improve how this is implemented
@dataclass
class EffectIsPendingInputTargets:
    entity_id: int
    name: str
    number_of_targets: int


@dataclass
class CombatView:
    character: Character
    monsters: list[Monster]
    hand: list[Card]
    draw_pile: set[Card]
    discard_pile: set[Card]
    energy: Energy
    effect: Optional[EffectIsPendingInputTargets]


def effect_is_pending_input_targets_view(
    manager: ECSManager,
) -> Optional[EffectIsPendingInputTargets]:
    try:
        effect_entity_id, (_, name_component, effect_number_of_targets_component) = next(
            manager.get_components(
                EffectIsPendingInputTargetsComponent, NameComponent, EffectNumberOfTargetsComponent
            )
        )
        return EffectIsPendingInputTargets(
            effect_entity_id, name_component.value, effect_number_of_targets_component.value
        )

    except StopIteration:
        return None


def character_view(manager: ECSManager) -> Character:
    entity_id, (_, name_component, health_component, block_component) = next(
        manager.get_components(CharacterComponent, NameComponent, HealthComponent, BlockComponent)
    )

    return Character(
        entity_id,
        name_component.value,
        Health(health_component.max, health_component.current),
        Block(block_component.max, block_component.current),
        modifiers=None,  # TODO: enable
    )


# TODO: list comprehension
def monsters_view(manager: ECSManager) -> list[Monster]:
    monsters = []
    for entity_id, (
        _,
        name_component,
        health_component,
        block_component,
    ) in manager.get_components(MonsterComponent, NameComponent, HealthComponent, BlockComponent):
        # Get the monster's current move
        intent_component = None
        for move_entity_id in manager.get_component_for_entity(
            entity_id, MonsterHasMovesComponent
        ).move_entity_ids:
            if (
                manager.get_component_for_entity(move_entity_id, MonsterCurrentMoveComponent)
                is not None
            ):
                intent_component = manager.get_component_for_entity(
                    move_entity_id, MonsterMoveIntentComponent
                )
                break

        # Modifiers
        actor_has_modifiers_component = manager.get_component_for_entity(
            entity_id, ActorHasModifiersComponent
        )
        modifiers = []
        if actor_has_modifiers_component is not None:
            for modifier_entity_id in actor_has_modifiers_component.modifier_entity_ids:
                name = manager.get_component_for_entity(modifier_entity_id, NameComponent).value
                modifier_stacks_component = manager.get_component_for_entity(
                    modifier_entity_id, ModifierStacksComponent
                )
                if modifier_stacks_component is not None:
                    stacks = modifier_stacks_component.value

                else:
                    stacks = None

                modifiers.append(Modifier(modifier_entity_id, name, stacks))

        monsters.append(
            Monster(
                entity_id,
                name_component.value,
                Health(health_component.max, health_component.current),
                Block(block_component.max, block_component.current),
                modifiers,
                (
                    False
                    if manager.get_component_for_entity(entity_id, CanBeSelectedComponent) is None
                    else True
                ),
                intent=(
                    Intent(intent_component.damage, intent_component.times, intent_component.block)
                    if intent_component is not None
                    else None
                ),
            )
        )

    # Sort according to position. TODO: unnecessary query
    monsters.sort(
        key=lambda monster: manager.get_component_for_entity(
            monster.entity_id, MonsterComponent
        ).position
    )

    return monsters


# TODO: list comprehension
def hand_view(manager: ECSManager) -> list[Card]:
    hand = []
    for entity_id, (
        _,
        name_component,
        description_component,
        card_cost_component,
    ) in manager.get_components(
        CardInHandComponent, NameComponent, DescriptionComponent, CardCostComponent
    ):
        hand.append(
            Card(
                entity_id,
                name_component.value,
                description_component.value,
                card_cost_component.value,
                (
                    False
                    if manager.get_component_for_entity(entity_id, IsSelectedComponent) is None
                    else True
                ),
                (
                    False
                    if manager.get_component_for_entity(entity_id, CanBeSelectedComponent) is None
                    else True
                ),
            )
        )

    # Sort according to position. TODO: unnecessary query
    hand.sort(
        key=lambda card: manager.get_component_for_entity(
            card.entity_id, CardInHandComponent
        ).position
    )

    return hand


def draw_pile_view(manager: ECSManager) -> list[Card]:
    return [
        Card(
            entity_id,
            name_component.value,
            description_component.value,
            card_cost_component.value,
            (
                False
                if manager.get_component_for_entity(entity_id, IsSelectedComponent) is None
                else True
            ),
            (
                False
                if manager.get_component_for_entity(entity_id, CanBeSelectedComponent) is None
                else True
            ),
        )
        for entity_id, (
            _,
            name_component,
            description_component,
            card_cost_component,
        ) in manager.get_components(
            CardInDrawPileComponent, NameComponent, DescriptionComponent, CardCostComponent
        )
    ]


def discard_pile_view(manager: ECSManager) -> list[Card]:
    return [
        Card(
            entity_id,
            name_component.value,
            description_component.value,
            card_cost_component.value,
            (
                False
                if manager.get_component_for_entity(entity_id, IsSelectedComponent) is None
                else True
            ),
            (
                False
                if manager.get_component_for_entity(entity_id, CanBeSelectedComponent) is None
                else True
            ),
        )
        for entity_id, (
            _,
            name_component,
            description_component,
            card_cost_component,
        ) in manager.get_components(
            CardInDiscardPileComponent, NameComponent, DescriptionComponent, CardCostComponent
        )
    ]


def energy_view(manager: ECSManager) -> Energy:
    entity_id, energy_component = next(manager.get_component(EnergyComponent))

    return Energy(entity_id, energy_component.max, energy_component.current)


def combat_view(manager: ECSManager) -> CombatView:
    return CombatView(
        character=character_view(manager),
        monsters=monsters_view(manager),
        hand=hand_view(manager),
        draw_pile=draw_pile_view(manager),
        discard_pile=discard_pile_view(manager),
        energy=energy_view(manager),
        effect=effect_is_pending_input_targets_view(manager),
    )
