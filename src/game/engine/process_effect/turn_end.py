from dataclasses import replace

from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.actor import EntityActor
from src.game.entity.actor import ModifierType
from src.game.entity.manager import EntityManager
from src.game.entity.monster import EntityMonster


def _get_end_of_turn_effects_common(actor: EntityActor) -> list[Effect]:
    effects = []
    for modifier_type, modifier_data in actor.modifier_map.items():  # TODO: revisit
        if modifier_type == ModifierType.RITUAL and not modifier_data.is_new:
            effects.append(
                Effect(
                    EffectType.MODIFIER_STRENGTH_GAIN,
                    modifier_data.stacks_current,
                    source=actor,
                    target=actor,
                )
            )

    return effects


def process_effect_turn_end(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    target = kwargs["target"]

    # Common effects (for both the Character and Monsters)
    effects = _get_end_of_turn_effects_common(target)

    if isinstance(target, EntityMonster):
        # Return end of turn effects to be added to the top so they are processed right away
        return [], effects

    # Discard all cards in the hand
    effects += [Effect(EffectType.CARD_DISCARD, target_type=EffectTargetType.CARD_IN_HAND)]

    # Set all modifiers as not new (so they can be decreased by `EffectType.MODIFIER_TICK`)
    effects.append(Effect(EffectType.MODIFIER_SET_NOT_NEW))

    # Queue effects for all monsters' turns
    for monster in entity_manager.monsters:
        # Turn start
        effects += [Effect(EffectType.TURN_START, target=monster)]

        # Move's effects
        effects += [
            replace(effect, source=monster)
            for effect in monster.moves[monster.move_name_current].effects
        ]

        # Update move
        effects += [Effect(EffectType.MONSTER_MOVE_UPDATE, target=monster)]

        # Turn end
        effects += [Effect(EffectType.TURN_END, target=monster)]

    # Character's turn start
    effects += [Effect(EffectType.TURN_START, target=entity_manager.character)]

    # Clear `ModifierType.BURST` TODO: here?
    if ModifierType.BURST in target.modifier_map:
        del target.modifier_map[ModifierType.BURST]

    # If the target is the Character, the queue should be empty at this point, so adding to the top
    # or to the bottom is the same
    return effects, []
