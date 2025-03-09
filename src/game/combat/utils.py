from src.game.combat.entities import Entities


def is_game_over(entities: Entities) -> bool:
    character = entities.all[entities.character_id]
    monsters = [entities.all[monster_id] for monster_id in entities.monster_ids]

    return character.health_current <= 0 or all(
        [monster.health_current <= 0 for monster in monsters]
    )
