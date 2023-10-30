from game.battle.pipelines.base import BasePipeline
from game.battle.pipelines.steps.deal_damage import DealDamage
from game.battle.pipelines.steps.gain_block import GainBlock


class CharacterPipeline(BasePipeline):
    _steps = [DealDamage(), GainBlock()]
