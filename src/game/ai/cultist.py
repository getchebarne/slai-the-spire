from src.game.entity.monster import EntityMonster


def get_move_name_cultist(monster: EntityMonster) -> str:
    if monster.move_name_current is None:
        return "Incantation"

    return "Dark Strike"
