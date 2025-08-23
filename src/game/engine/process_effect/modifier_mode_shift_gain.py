from src.game.core.effect import Effect
from src.game.entity.actor import ModifierData
from src.game.entity.actor import ModifierType
from src.game.entity.manager import EntityManager
from src.game.factory.monster.the_guardian import _MODE_SHIFT_STACKS_CURRENT
from src.game.factory.monster.the_guardian import _MODE_SHIFT_STACKS_CURRENT_ASC_9
from src.game.factory.monster.the_guardian import _MODE_SHIFT_STACKS_CURRENT_ASC_19
from src.game.factory.monster.the_guardian import _MODE_SHIFT_STACKS_DURATION
from src.game.factory.monster.the_guardian import _MODE_SHIFT_STACKS_MAX
from src.game.factory.monster.the_guardian import _MODE_SHIFT_STACKS_MIN


IS_BUFF = True
_STACK_INCREASE_PER_CYCLE = 10


def process_effect_modifier_mode_shift_gain(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    id_target = kwargs["id_target"]
    ascension_level = kwargs["ascension_level"]

    target = entity_manager.entities[id_target]

    # Count number of cycles
    cycle_num = target.move_name_history.count("Twin Slam")
    stack_increase = _STACK_INCREASE_PER_CYCLE * cycle_num

    if ascension_level < 9:
        stacks = _MODE_SHIFT_STACKS_CURRENT + stack_increase

    elif ascension_level < 19:
        stacks = _MODE_SHIFT_STACKS_CURRENT_ASC_9 + stack_increase

    else:
        stacks = _MODE_SHIFT_STACKS_CURRENT_ASC_19 + stack_increase

    if ModifierType.MODE_SHIFT in target.modifier_map:
        raise ValueError("TODO: add message")

    target.modifier_map[ModifierType.MODE_SHIFT] = ModifierData(
        IS_BUFF,
        False,
        min(stacks, _MODE_SHIFT_STACKS_MAX),
        _MODE_SHIFT_STACKS_MIN,
        _MODE_SHIFT_STACKS_MAX,
        _MODE_SHIFT_STACKS_DURATION,
    )

    return [], []
