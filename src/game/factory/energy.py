from src.game.entity.energy import EntityEnergy


def create_energy(max_: int, current: int) -> EntityEnergy:
    return EntityEnergy(max_, current)
