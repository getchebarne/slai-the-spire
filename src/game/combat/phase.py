from src.game.combat.effect import Effect
from src.game.combat.effect import EffectTargetType
from src.game.combat.effect import EffectType
from src.game.entity.actor import ModifierType
from src.game.entity.character import EntityCharacter
from src.game.entity.manager import EntityManager


def get_start_of_combat_effects(entity_manager: EntityManager) -> list[Effect]:
    return [
        Effect(EffectType.CARD_SHUFFLE_DECK_INTO_DRAW_PILE),
        *[
            Effect(EffectType.MONSTER_MOVE_UPDATE, id_target=id_monster)
            for id_monster in entity_manager.id_monsters
        ],
    ] + get_start_of_turn_effects(entity_manager, entity_manager.id_character)


# TODO: add modifier effects
def get_start_of_turn_effects(entity_manager: EntityManager, id_actor: int) -> list[Effect]:
    actor = entity_manager.entities[id_actor]

    # Common effects
    effects = [Effect(EffectType.BLOCK_RESET, id_target=id_actor)]

    # Character-specific effects
    if isinstance(actor, EntityCharacter):
        energy = entity_manager.entities[entity_manager.id_energy]

        effects += [
            Effect(EffectType.CARD_DRAW, 5),
            Effect(EffectType.ENERGY_GAIN, energy.max - energy.current),
            Effect(EffectType.MODIFIER_TICK, id_target=id_actor),
        ] + [
            Effect(EffectType.MODIFIER_TICK, id_target=id_monster)
            for id_monster in entity_manager.id_monsters
        ]

    return effects


# TODO: add modifier effects
def get_end_of_turn_effects(entity_manager: EntityManager, id_actor: int) -> list[Effect]:
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

    # Character-specific effects
    if isinstance(actor, EntityCharacter):
        return effects + [
            Effect(EffectType.CARD_DISCARD, target_type=EffectTargetType.CARD_IN_HAND)
        ]

    return effects
