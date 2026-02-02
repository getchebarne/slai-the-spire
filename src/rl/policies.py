"""
Policy implementations for action selection.

Available policies:
- PolicyBase: Abstract base class for all policies
- PolicyRandom: Random action selection (useful for baselines and exploration)

Note: The PolicySoftmax class requires the new hierarchical ActorCritic model.
For model-based action selection, use ActorCritic.get_action() directly.
"""

import random
from abc import ABC, abstractmethod
from typing import Any, TypeAlias

from src.game.action import Action
from src.game.action import ActionType
from src.game.view.fsm import ViewFSM
from src.game.view.state import ViewGameState


SelectActionMetadata: TypeAlias = dict[str, Any]


class PolicyBase(ABC):
    """Abstract base class for all policies."""

    @abstractmethod
    def select_action(self, view_game_state: ViewGameState) -> tuple[Action, SelectActionMetadata]:
        """
        Select an action given the current game state.

        Args:
            view_game_state: Current game state view

        Returns:
            Tuple of (Action to take, metadata dict for logging/debugging)
        """
        raise NotImplementedError


class PolicyRandom(PolicyBase):
    """
    Random policy that selects valid actions uniformly at random.

    Useful for:
    - Establishing baseline performance
    - Exploration during training
    - Testing game mechanics
    """

    def select_action(self, view_game_state: ViewGameState) -> tuple[Action, SelectActionMetadata]:
        """Select a random valid action."""

        # Card Reward screen
        if view_game_state.fsm == ViewFSM.CARD_REWARD:
            num_cards = len(view_game_state.reward_combat)
            roll = random.randint(0, num_cards)
            if roll == num_cards:
                return Action(ActionType.CARD_REWARD_SKIP), {}
            return Action(ActionType.CARD_REWARD_SELECT, roll), {}

        # Map screen
        if view_game_state.fsm == ViewFSM.MAP:
            if view_game_state.map.y_current is None:
                # First floor - select from starting nodes
                valid_x = [
                    x for x, node in enumerate(view_game_state.map.nodes[0]) if node is not None
                ]
                return Action(ActionType.MAP_NODE_SELECT, random.choice(valid_x)), {}

            # Subsequent floors - select from connected nodes
            map_node = view_game_state.map.nodes[view_game_state.map.y_current][
                view_game_state.map.x_current
            ]
            return (
                Action(ActionType.MAP_NODE_SELECT, random.choice(list(map_node.x_next))),
                {},
            )

        # Rest site
        if view_game_state.fsm == ViewFSM.REST_SITE:
            action_type = random.choice([ActionType.REST_SITE_REST, ActionType.REST_SITE_UPGRADE])
            if action_type == ActionType.REST_SITE_REST:
                return Action(action_type), {}

            # Find upgradable cards
            index_upgradable = [
                idx for idx, card in enumerate(view_game_state.deck) if not card.name.endswith("+")
            ]
            if not index_upgradable:
                # No upgradable cards, rest instead
                return Action(ActionType.REST_SITE_REST), {}

            return Action(action_type, random.choice(index_upgradable)), {}

        # Combat - awaiting monster target
        if view_game_state.fsm == ViewFSM.COMBAT_AWAIT_TARGET_CARD:
            return (
                Action(
                    ActionType.COMBAT_MONSTER_SELECT,
                    random.choice(range(len(view_game_state.monsters))),
                ),
                {},
            )

        # Combat - awaiting discard target
        if view_game_state.fsm == ViewFSM.COMBAT_AWAIT_TARGET_DISCARD:
            return (
                Action(
                    ActionType.COMBAT_CARD_IN_HAND_SELECT,
                    random.choice(range(len(view_game_state.hand))),
                ),
                {},
            )

        # Combat - default (play card or end turn)
        card_selectable_pos = [
            pos
            for pos, card in enumerate(view_game_state.hand)
            if card.cost <= view_game_state.energy.current
        ]
        if card_selectable_pos:
            return (
                Action(
                    ActionType.COMBAT_CARD_IN_HAND_SELECT,
                    random.choice(card_selectable_pos),
                ),
                {},
            )

        return Action(ActionType.COMBAT_TURN_END), {}
