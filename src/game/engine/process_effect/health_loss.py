from src.game.combat.effect import Effect
from src.game.combat.effect import EffectTargetType
from src.game.combat.effect import EffectType
from src.game.entity.actor import ModifierType
from src.game.entity.manager import EntityManager
from src.game.entity.monster import EntityMonster


def process_effect_health_loss(
    entity_manager: EntityManager, effect: Effect
) -> tuple[list[Effect], list[Effect]]:
    target = entity_manager.entities[effect.id_target]

    if effect.value >= target.health_current:
        # Death
        if isinstance(target, EntityMonster):
            # TODO: delete instance from `entity_manager.entities`
            entity_manager.id_monsters.remove(effect.id_target)

            effects_top = []
            for modifier_type, modifier_data in target.modifier_map.items():
                if modifier_type == ModifierType.SPORE_CLOUD:
                    effects_top.append(
                        Effect(
                            EffectType.MODIFIER_VULNERABLE_GAIN,
                            modifier_data.stacks_current,
                            EffectTargetType.CHARACTER,
                        )
                    )

            return [], effects_top

    target.health_current = max(0, target.health_current - effect.value)

    return [], []
