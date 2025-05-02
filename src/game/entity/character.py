from dataclasses import dataclass

from src.game.entity.actor import EntityActor


_CARD_REWARD_ROLL_OFFSET_BASE = 5


@dataclass
class EntityCharacter(EntityActor):
    card_reward_roll_offset: int = _CARD_REWARD_ROLL_OFFSET_BASE
