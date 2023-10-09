from dataclasses import dataclass

from components.base import BaseComponent


@dataclass
class Intent(BaseComponent):
    damage: int
    instances: int
    block: bool
    buff: bool
    debuff: bool
    strong_debuff: bool
    escape: bool
    asleep: bool
    stunned: bool
    unknown: bool
