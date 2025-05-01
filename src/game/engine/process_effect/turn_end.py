from dataclasses import replace

from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.actor import ModifierType
from src.game.entity.manager import EntityManager
from src.game.entity.monster import EntityMonster


def _get_end_of_turn_effects_common(entity_manager: EntityManager, id_actor: int) -> list[Effect]:
    actor = entity_manager.entities[id_actor]

    effects = []
    for modifier_type, modifier_data in actor.modifier_map.items():
        if modifier_type == ModifierType.RITUAL:
            effects.append(
                Effect(
                    EffectType.MODIFIER_STRENGTH_GAIN,
                    modifier_data.stacks_current,
                    id_source=id_actor,
                    id_target=id_actor,
                )
            )

    return effects


def process_effect_turn_end(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    id_target = kwargs["id_target"]

    target = entity_manager.entities[id_target]

    # Common effects (for both the Character and Monsters)
    effects = _get_end_of_turn_effects_common(entity_manager, id_target)

    if isinstance(target, EntityMonster):
        # Return end of turn effects to be added to the top so they are processed right away
        return [], effects

    # Discard all cards in the hand
    effects += [Effect(EffectType.CARD_DISCARD, target_type=EffectTargetType.CARD_IN_HAND)]

    # Queue effects for all monsters' turns
    for id_monster in entity_manager.id_monsters:
        monster = entity_manager.entities[id_monster]

        # Turn start
        effects += [Effect(EffectType.TURN_START, id_target=id_monster)]

        # Move's effects
        effects += [
            replace(effect, id_source=id_monster)
            for effect in monster.move_map[monster.move_name_current]
        ]

        # Update move
        effects += [Effect(EffectType.MONSTER_MOVE_UPDATE, id_target=id_monster)]

        # Turn end
        effects += [Effect(EffectType.TURN_END, id_target=id_monster)]

    # Character's turn start
    effects += [Effect(EffectType.TURN_START, id_target=entity_manager.id_character)]

    # If the target is the Character, the queue should be empty at this point, so adding to the top
    # or to the bottom is the same
    return effects, []
