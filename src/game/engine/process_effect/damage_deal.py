from src.game.combat.effect import Effect
from src.game.combat.effect import EffectType
from src.game.entity.actor import ModifierType
from src.game.entity.card import EntityCard
from src.game.entity.manager import EntityManager


FACTOR_WEAK = 0.75
FACTOR_VULN = 1.50


def process_effect_damage_deal(
    entity_manager: EntityManager, effect: Effect
) -> tuple[list[Effect], list[Effect]]:
    source = entity_manager.entities[effect.id_source]
    target = entity_manager.entities[effect.id_target]

    # TODO: think if there's a better solution
    if isinstance(source, EntityCard):
        source = entity_manager.entities[entity_manager.id_character]

    # Apply strength
    value = effect.value
    if ModifierType.STRENGTH in source.modifier_map:
        value += source.modifier_map[ModifierType.STRENGTH].stacks_current

    # Apply weak
    if ModifierType.WEAK in source.modifier_map:
        value *= FACTOR_WEAK

    # Apply vulnerable
    if ModifierType.VULNERABLE in target.modifier_map:
        value *= FACTOR_VULN

    # Calculate damage over block
    value = int(value)
    damage_over_block = max(0, value - target.block_current)

    # Remove block
    target.block_current = max(0, target.block_current - value)

    if damage_over_block > 0:
        # Return a top effect to subtract the damage over block from the target's current health
        return [], [
            Effect(EffectType.HEALTH_LOSS, value=damage_over_block, id_target=effect.id_target)
        ]

    return [], []
