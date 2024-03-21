from dataclasses import dataclass


@dataclass
class Intent:
    attack: bool = False
    block: bool = False
    buff: bool = False
    debuff: bool = False
    strong_debuff: bool = False
    escape: bool = False
    stunned: bool = False
    asleep: bool = False
    unknown: bool = False

    # Used for attack intents only
    attack_damage: int = 0
    attack_instances: int = 0
