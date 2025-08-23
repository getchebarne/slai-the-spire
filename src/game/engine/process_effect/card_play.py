from dataclasses import replace

from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.actor import ModifierType
from src.game.entity.card import CardType
from src.game.entity.manager import EntityManager


def process_effect_card_play(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    id_target = kwargs["id_target"]

    target = entity_manager.entities[id_target]

    # Energy loss
    effects_top = [Effect(EffectType.ENERGY_LOSS, value=target.cost)]

    # Exhaust vs. power vs. discard
    if target.exhaust:
        effects_top.append(Effect(EffectType.CARD_EXHAUST, id_target=id_target))
    elif target.type == CardType.POWER:
        effects_top.append(Effect(EffectType.CARD_REMOVE, id_target=id_target))
    else:
        effects_top.append(Effect(EffectType.CARD_DISCARD, id_target=id_target))

    # After image
    character = entity_manager.entities[entity_manager.id_character]
    if ModifierType.AFTER_IMAGE in character.modifier_map:
        stacks_current = character.modifier_map[ModifierType.AFTER_IMAGE].stacks_current
        effects_top.append(
            Effect(
                EffectType.BLOCK_GAIN,
                stacks_current,
                id_target=entity_manager.id_character,
                id_source=entity_manager.id_character,
            )
        )

    # Thousand cuts
    if ModifierType.THOUSAND_CUTS in character.modifier_map:
        stacks_current = character.modifier_map[ModifierType.THOUSAND_CUTS].stacks_current
        effects_top.append(
            Effect(
                EffectType.DAMAGE_DEAL_PHYSICAL,
                stacks_current,
                target_type=EffectTargetType.MONSTER,
                id_source=entity_manager.id_character,
            )
        )

    # Card's effects
    effects_top.extend([replace(effect, id_source=id_target) for effect in target.effects])
    if ModifierType.BURST in character.modifier_map and target.type == CardType.SKILL:
        effects_top.extend([replace(effect, id_source=id_target) for effect in target.effects])
        effects_top.append(
            Effect(EffectType.MODIFIER_BURST_LOSS, 1, id_target=entity_manager.id_character)
        )

    # Sharp hide
    if entity_manager.id_card_target is not None:
        target_card = entity_manager.entities[entity_manager.id_card_target]
        if ModifierType.SHARP_HIDE in target_card.modifier_map and target.type == CardType.ATTACK:
            modifier_data = target_card.modifier_map[ModifierType.SHARP_HIDE]
            effects_top.append(
                Effect(
                    EffectType.DAMAGE_DEAL,
                    value=modifier_data.stacks_current,
                    id_source=entity_manager.id_card_target,
                    id_target=entity_manager.id_character,
                )
            )

    return [], effects_top
