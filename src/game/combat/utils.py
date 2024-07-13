from src.game.combat.entities import Entities


def is_game_over(entities: Entities) -> bool:
    return entities.get_character().health.current <= 0 or all(
        [monster.health.current <= 0 for monster in entities.get_monsters()]
    )
