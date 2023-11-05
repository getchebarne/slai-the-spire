from game.battle.pipeline.base import BasePipeline
from game.battle.pipeline.steps.deal_damage import DealDamage
from game.battle.pipeline.steps.gain_block import GainBlock


class CharacterPipeline(BasePipeline):
    _steps = [DealDamage(), GainBlock()]
