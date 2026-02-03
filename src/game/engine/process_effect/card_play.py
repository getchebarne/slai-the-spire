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
    target = kwargs["target"]  # The card being played

    # Energy loss
    effects_top = [Effect(EffectType.ENERGY_LOSS, value=target.cost)]

    # Exhaust vs. power vs. discard
    if target.exhaust:
        effects_top.append(Effect(EffectType.CARD_EXHAUST, target=target))
    elif target.type == CardType.POWER:
        effects_top.append(Effect(EffectType.CARD_REMOVE, target=target))
    else:
        effects_top.append(Effect(EffectType.CARD_DISCARD, target=target))

    # After image
    character = entity_manager.character
    if ModifierType.AFTER_IMAGE in character.modifier_map:
        stacks_current = character.modifier_map[ModifierType.AFTER_IMAGE].stacks_current
        effects_top.append(
            Effect(
                EffectType.BLOCK_GAIN,
                stacks_current,
                target=character,
                source=character,
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
                source=character,
            )
        )

    # Card's effects
    effects_top.extend([replace(effect, source=target) for effect in target.effects])
    if ModifierType.BURST in character.modifier_map and target.type == CardType.SKILL:
        effects_top.extend([replace(effect, source=target) for effect in target.effects])
        effects_top.append(Effect(EffectType.MODIFIER_BURST_LOSS, 1, target=character))

    # Sharp hide (target monster has sharp hide, damages character when attacked)
    for monster in entity_manager.monsters:
        if ModifierType.SHARP_HIDE in monster.modifier_map and target.type == CardType.ATTACK:
            modifier_data = monster.modifier_map[ModifierType.SHARP_HIDE]
            effects_top.append(
                Effect(
                    EffectType.DAMAGE_DEAL,
                    value=modifier_data.stacks_current,
                    source=monster,
                    target=character,
                )
            )

    return [], effects_top
