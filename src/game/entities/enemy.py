from dataclasses import dataclass
from typing import Literal

from components.health import Health
from components.intent import Intent
from entities.base import BaseEntity


ALL_ENEMIES = Literal["Jaw Worm", "Cultist"]


@dataclass
class Enemy(BaseEntity):
    health: Health
    intent: Intent
    name: ALL_ENEMIES
