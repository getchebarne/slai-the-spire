from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.entity.actor import ModifierType
from src.game.entity.card import EntityCard
from src.game.entity.manager import EntityManager


FACTOR_WEAK = 0.75
FACTOR_VULN = 1.50


def process_effect_damage_deal_physical(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    value = kwargs["value"]
    id_source = kwargs["id_source"]
    id_target = kwargs["id_target"]

    source = entity_manager.entities[id_source]
    target = entity_manager.entities[id_target]

    # TODO: think if there's a better solution
    if isinstance(source, EntityCard):
        source = entity_manager.entities[entity_manager.id_character]

    # Apply strength
    value = value
    if ModifierType.STRENGTH in source.modifier_map:
        value += source.modifier_map[ModifierType.STRENGTH].stacks_current

    # Apply weak
    if ModifierType.WEAK in source.modifier_map:
        value *= FACTOR_WEAK

    # Apply vulnerable
    if ModifierType.VULNERABLE in target.modifier_map:
        value *= FACTOR_VULN

    # Calculate damage
    value = int(value)

    if value > 0:
        # Return a top effect to subtract the damage over block from the target's current health
        return [], [Effect(EffectType.DAMAGE_DEAL, value, id_target=id_target)]

    return [], []
