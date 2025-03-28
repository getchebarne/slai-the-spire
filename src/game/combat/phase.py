from dataclasses import dataclass

from src.game.combat.entities import Character
from src.game.combat.entities import Effect
from src.game.combat.entities import EffectTargetType
from src.game.combat.entities import EffectType
from src.game.combat.entities import EntityManager


# TODO: move to common dataclasses or sth script, processors.py also uses this
@dataclass(frozen=True)
class ToBeQueuedEffect:
    effect: Effect
    id_source: int | None = None
    id_target: int | None = None


def get_start_of_combat_effects(entity_manager: EntityManager) -> list[ToBeQueuedEffect]:
    return [
        ToBeQueuedEffect(Effect(EffectType.SHUFFLE_DECK_INTO_DRAW_PILE)),
        *[
            ToBeQueuedEffect(Effect(EffectType.UPDATE_MOVE), id_target=id_monster)
            for id_monster in entity_manager.id_monsters
        ],
    ] + get_start_of_turn_effects(entity_manager, entity_manager.id_character)


# TODO: add modifier effects
def get_start_of_turn_effects(
    entity_manager: EntityManager, id_actor: int
) -> list[ToBeQueuedEffect]:
    actor = entity_manager.entities[id_actor]

    # Common effects
    to_be_queued_effects = [
        ToBeQueuedEffect(
            Effect(EffectType.ZERO_BLOCK, target_type=EffectTargetType.SOURCE), id_source=id_actor
        )
    ]

    # Character-specific effects
    if isinstance(actor, Character):
        to_be_queued_effects += [
            ToBeQueuedEffect(Effect(EffectType.DRAW_CARD, 5)),
            ToBeQueuedEffect(Effect(EffectType.REFILL_ENERGY)),
        ]

    return to_be_queued_effects


# TODO: add modifier effects
# TODO: add modifier duration tick (`EffectType.MOD_TICK`)
def get_end_of_turn_effects(
    entity_manager: EntityManager, id_actor: int
) -> list[ToBeQueuedEffect]:
    actor = entity_manager.entities[id_actor]

    # Character-specific effects
    if isinstance(actor, Character):
        return [
            ToBeQueuedEffect(Effect(EffectType.DISCARD, target_type=EffectTargetType.CARD_IN_HAND))
        ]

    return []
