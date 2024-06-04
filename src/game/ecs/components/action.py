from dataclasses import dataclass
from typing import Optional

from src.game.combat.action import ActionType
from src.game.ecs.components.base import BaseComponent


@dataclass
class ActionComponent(BaseComponent):
    type: ActionType
    target_entity_id: Optional[int]
