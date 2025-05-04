from src.game.entity.actor import ModifierType
from src.game.entity.monster import EntityMonster


def get_move_name_the_guardian(monster: EntityMonster) -> str:
    if monster.move_name_current is None:
        return "Charging Up"

    if ModifierType.MODE_SHIFT in monster.modifier_map:
        # Offensive mode
        if monster.move_name_history[-1] == "Charging Up":
            return "Fierce Bash"

        if monster.move_name_history[-1] == "Fierce Bash":
            return "Vent Steam"

        if monster.move_name_history[-1] == "Vent Steam":
            return "Whirlwind"

        if monster.move_name_history[-1] == "Whirlwind":
            return "Charging Up"

        if monster.move_name_history[-1] == "Twin Slam":
            return "Whirlwind"

        raise ValueError(f"Unsupported move name: {monster.move_name_history[-1]}")

    # Defensive mode
    if ModifierType.SHARP_HIDE in monster.modifier_map:
        if monster.move_name_history[-1] == "Defensive Mode":
            return "Roll Attack"

        if monster.move_name_history[-1] == "Roll Attack":
            return "Twin Slam"

        raise ValueError(f"Unsupported move name: {monster.move_name_history[-1]}")

    return "Defensive Mode"
