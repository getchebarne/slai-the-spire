from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.entity.actor import ModifierType
from src.game.entity.card import EntityCard
from src.game.entity.manager import EntityManager


FACTOR_WEAK = 0.75
FACTOR_VULN = 1.50
_SHIV_NAME = "Shiv"  # TODO: improve this crap


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
        character = entity_manager.entities[entity_manager.id_character]

        # Apply accuracy bonus damage
        if (
            source.name == _SHIV_NAME or source.name == f"{_SHIV_NAME}+"
        ) and ModifierType.ACCURACY in character.modifier_map:
            value += character.modifier_map[ModifierType.ACCURACY].stacks_current

        # Overwrite source with character to apply modifiers
        source = entity_manager.entities[entity_manager.id_character]

    # Apply strength
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
        return [], [
            Effect(EffectType.DAMAGE_DEAL, value, id_target=id_target, id_source=id_source)
        ]

    return [], []
