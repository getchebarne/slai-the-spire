from dataclasses import dataclass, field

from src.game.core.effect import Effect
from src.game.entity.actor import EntityActor


@dataclass
class EntityMonster(EntityActor):
    move_map: dict[str, list[Effect]] = field(default_factory=dict)
    move_name_current: str | None = None
    move_name_history: list[str] = field(default_factory=list)
