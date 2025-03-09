from dataclasses import dataclass
from typing import NamedTuple

from src.game.combat.entities import Effect
from src.game.combat.entities import Entities


# TODO: revisit name
class QueuedEffect(NamedTuple):
    effect: Effect
    source_id: int | None = None
    target_id: int | None = None


def add_to_bot(
    effect_queue: list[QueuedEffect], source_id: int, *effects: Effect
) -> list[QueuedEffect]:
    return effect_queue + [QueuedEffect(effect, source_id) for effect in effects]


def add_to_top(
    effect_queue: list[QueuedEffect], source_id: int, *effects: Effect
) -> list[QueuedEffect]:
    return [QueuedEffect(effect, source_id) for effect in effects] + effect_queue


@dataclass
class CombatState:
    entities: Entities
    effect_queue: list[QueuedEffect]
