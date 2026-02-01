"""
Neural network models for reinforcement learning.

Main exports:
- ActorCritic: The main model combining encoder and action heads
- Core: The shared encoder (entity transformer + map encoder)
- heads: All action heads (HeadActionType, HeadValue, entity selection heads, etc.)
"""

from src.rl.models.actor_critic import ActorCritic
from src.rl.models.actor_critic import ActorCriticOutput
from src.rl.models.actor_critic import ActorOutput
from src.rl.models.actor_critic import BatchedActorCriticOutput
from src.rl.models.actor_critic import BatchedActorOutput
from src.rl.models.core import Core
from src.rl.models.core import CoreOutput
from src.rl.models.heads import HeadActionType
from src.rl.models.heads import HeadCardDiscard
from src.rl.models.heads import HeadCardPlay
from src.rl.models.heads import HeadCardRewardSelect
from src.rl.models.heads import HeadCardUpgrade
from src.rl.models.heads import HeadEntitySelection
from src.rl.models.heads import HeadMapSelect
from src.rl.models.heads import HeadMonsterSelect
from src.rl.models.heads import HeadOutput
from src.rl.models.heads import HeadValue


__all__ = [
    # Actor-Critic model
    "ActorCritic",
    "ActorCriticOutput",
    "ActorOutput",
    "BatchedActorCriticOutput",
    "BatchedActorOutput",
    # Core encoder
    "Core",
    "CoreOutput",
    # Heads
    "HeadEntitySelection",
    "HeadActionType",
    "HeadCardDiscard",
    "HeadCardPlay",
    "HeadCardRewardSelect",
    "HeadCardUpgrade",
    "HeadMapSelect",
    "HeadMonsterSelect",
    "HeadOutput",
    "HeadValue",
]
