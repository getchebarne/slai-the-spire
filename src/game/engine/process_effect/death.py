from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.actor import ModifierType
from src.game.entity.character import EntityCharacter
from src.game.entity.manager import EntityManager
from src.game.entity.monster import EntityMonster


def process_effect_death(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    id_target = kwargs["id_target"]

    target = entity_manager.entities[id_target]
    if isinstance(target, EntityCharacter):
        # Game's over
        return [], [Effect(EffectType.GAME_END)]

    if isinstance(target, EntityMonster):
        # TODO: delete instance from `entity_manager.entities`
        entity_manager.id_monsters.remove(id_target)

        if entity_manager.id_monsters:
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

        # Combat's over
        return [], [Effect(EffectType.COMBAT_END)]

    raise ValueError(f"Unsupported entity type: {type(target)}")
