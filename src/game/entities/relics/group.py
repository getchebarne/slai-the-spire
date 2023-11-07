from game.battle.pipeline.pipeline import EffectPipeline
from game.entities.relics.base import BaseRelic


class RelicGroup(list):
    def __init__(self):
        super().__init__([])

    # TODO: should this be moved to the `battle` module?
    def add_relic(self, relic: BaseRelic, pipeline: EffectPipeline) -> None:
        if not isinstance(relic, BaseRelic):
            raise TypeError(
                f"Can only add objects of {BaseRelic} type. Received type: {type(relic)}"
            )

        # If the relic has a corresponding EffectPipeline step, add it
        if relic.step is not None:
            pipeline.add_step(relic.step)

        self.append(relic)
