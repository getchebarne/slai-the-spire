from src.game.core.effect import Effect
from src.game.core.effect import EffectSelectionType
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.actor import ModifierType
from src.game.entity.manager import EntityManager
from src.game.entity.monster import EntityMonster


def process_effect_health_loss(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    value = kwargs["value"]
    id_target = kwargs["id_target"]

    target = entity_manager.entities[id_target]
    if value >= target.health_current:
        # Death TODO: improve
        if isinstance(target, EntityMonster):
            # TODO: delete instance from `entity_manager.entities`
            entity_manager.id_monsters.remove(id_target)

            if not entity_manager.id_monsters:
                # Combat over
                return [], [
                    Effect(EffectType.COMBAT_END),
                    Effect(EffectType.CARD_REWARD_ROLL),
                    Effect(
                        EffectType.CARD_REWARD_SELECT,
                        target_type=EffectTargetType.CARD_REWARD,
                        selection_type=EffectSelectionType.INPUT,
                    ),
                    Effect(
                        EffectType.MAP_NODE_ACTIVE_SET,
                        target_type=EffectTargetType.MAP_NODE,
                        selection_type=EffectSelectionType.INPUT,
                    ),
                ]

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

    target.health_current = max(0, target.health_current - value)

    return [], []
