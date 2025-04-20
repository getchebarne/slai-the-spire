from src.game.entity.actor import ModifierData


STACKS_MIN = 1
STACKS_MAX = 999
STACKS_DURATION = False


def create_modifier_data_ritual(stacks_current: int) -> ModifierData:
    return ModifierData(stacks_current, STACKS_MIN, STACKS_MAX, STACKS_DURATION)
