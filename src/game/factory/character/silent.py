from src.game.entity.character import EntityCharacter


# TODO: add ascension starting health
def create_character_silent(health_current: int, health_max: int) -> EntityCharacter:
    return EntityCharacter("Silent", health_current=health_current, health_max=health_max)
