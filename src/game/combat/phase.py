from src.game.combat.effect import Effect
from src.game.combat.effect import EffectTargetType
from src.game.combat.effect import EffectType
from src.game.combat.effect import SourcedEffect
from src.game.combat.entities import Character
from src.game.combat.entities import EntityManager


def get_start_of_combat_effects(entity_manager: EntityManager) -> list[SourcedEffect]:
    return [
        SourcedEffect(Effect(EffectType.SHUFFLE_DECK_INTO_DRAW_PILE)),
        *[
            SourcedEffect(Effect(EffectType.UPDATE_MOVE), id_target=id_monster)
            for id_monster in entity_manager.id_monsters
        ],
    ] + get_start_of_turn_effects(entity_manager, entity_manager.id_character)


# TODO: add modifier effects
def get_start_of_turn_effects(entity_manager: EntityManager, id_actor: int) -> list[SourcedEffect]:
    actor = entity_manager.entities[id_actor]

    # Common effects
    sourced_effects = [SourcedEffect(Effect(EffectType.ZERO_BLOCK), id_target=id_actor)]

    # Character-specific effects
    if isinstance(actor, Character):
        sourced_effects += [
            SourcedEffect(Effect(EffectType.DRAW_CARD, 5)),
            SourcedEffect(Effect(EffectType.REFILL_ENERGY)),
        ]

    return sourced_effects


# TODO: add modifier effects
# TODO: add modifier duration tick (`EffectType.MOD_TICK`)
def get_end_of_turn_effects(entity_manager: EntityManager, id_actor: int) -> list[SourcedEffect]:
    actor = entity_manager.entities[id_actor]

    # Character-specific effects
    if isinstance(actor, Character):
        return [
            SourcedEffect(Effect(EffectType.DISCARD, target_type=EffectTargetType.CARD_IN_HAND))
        ]

    return []
