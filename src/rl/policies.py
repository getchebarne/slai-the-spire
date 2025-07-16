import random
from abc import ABC, abstractmethod
from typing import Any, TypeAlias

import torch
import torch.nn as nn

from src.game.action import Action
from src.game.action import ActionType
from src.game.view.fsm import ViewFSM
from src.game.view.state import ViewGameState


# from src.rl.encoding import encode_combat_view
# from src.rl.models.interface import action_idx_to_action
# from src.rl.models.interface import get_valid_action_mask


SelectActionMetadata: TypeAlias = dict[str, Any]


class PolicyBase(ABC):
    @abstractmethod
    def select_action(view_game_state: ViewGameState) -> tuple[Action, SelectActionMetadata]:
        raise NotImplementedError


class PolicyRandom(PolicyBase):
    def select_action(self, view_game_state: ViewGameState) -> tuple[Action, SelectActionMetadata]:
        if view_game_state.fsm == ViewFSM.CARD_REWARD:
            num_cards = len(view_game_state.reward_combat)
            roll = random.randint(0, num_cards)
            if roll == num_cards:
                return Action(ActionType.CARD_REWARD_SKIP), {}

            return Action(ActionType.CARD_REWARD_SELECT, roll), {}

        if view_game_state.fsm == ViewFSM.MAP:
            if view_game_state.map.y_current is None:
                return (
                    Action(
                        ActionType.MAP_NODE_SELECT,
                        random.choice(
                            [
                                x
                                for x, node in enumerate(view_game_state.map.nodes[0])
                                if node is not None
                            ]
                        ),
                    ),
                    {},
                )

            map_node = view_game_state.map.nodes[view_game_state.map.y_current][
                view_game_state.map.x_current
            ]
            return Action(ActionType.MAP_NODE_SELECT, random.choice(list(map_node.x_next))), {}

        if view_game_state.fsm == ViewFSM.REST_SITE:
            action_type = random.choice([ActionType.REST_SITE_REST, ActionType.REST_SITE_UPGRADE])
            if action_type == ActionType.REST_SITE_REST:
                return Action(action_type), {}

            index_upgradable = [
                idx for idx, card in enumerate(view_game_state.deck) if not card.name.endswith("+")
            ]
            return (
                Action(action_type, random.choice(index_upgradable)),
                {},
            )

        if view_game_state.fsm == ViewFSM.COMBAT_AWAIT_TARGET_CARD:
            return (
                Action(
                    ActionType.COMBAT_MONSTER_SELECT,
                    random.choice(range(len(view_game_state.monsters))),
                ),
                {},
            )

        if view_game_state.fsm == ViewFSM.COMBAT_AWAIT_TARGET_DISCARD is not None:
            return (
                Action(
                    ActionType.COMBAT_CARD_IN_HAND_SELECT,
                    random.choice(range(len(view_game_state.hand))),
                ),
                {},
            )

        card_selectable_pos = [
            pos
            for pos, card in enumerate(view_game_state.hand)
            if card.cost <= view_game_state.energy.current
        ]
        if card_selectable_pos:
            return (
                Action(ActionType.COMBAT_CARD_IN_HAND_SELECT, random.choice(card_selectable_pos)),
                {},
            )

        return Action(ActionType.COMBAT_TURN_END), {}


class PolicyQMax(PolicyBase):
    def __init__(self, model: nn.Module, device: torch.device):
        self._model = model
        self._device = device

        # Send model to device
        self._model.to(self._device)

    def select_action(self, view_game_state: ViewGameState) -> tuple[Action, SelectActionMetadata]:
        combat_view_encoding = encode_combat_view(combat_view, self._device)
        valid_action_mask_tensor = torch.tensor(
            [get_valid_action_mask(combat_view)], dtype=torch.bool, device=self._device
        )

        with torch.no_grad():
            q_values = self._model(*combat_view_encoding.as_tuple())

        q_values[~valid_action_mask_tensor] = float("-inf")
        action_idx = torch.argmax(q_values, dim=1).item()
        action = action_idx_to_action(action_idx, combat_view)

        return action, {"q_values": q_values}


class PolicySoftmax(PolicyBase):
    def __init__(self, model: nn.Module, device: torch.device, greedy: bool = True):
        self._model = model
        self._device = device
        self._greedy = greedy

        # Send model to device
        self._model.to(self._device)

    def select_action(self, view_game_state: ViewGameState) -> tuple[Action, SelectActionMetadata]:
        combat_view_encoding = encode_combat_view(combat_view, self._device)
        valid_action_mask_tensor = torch.tensor(
            [get_valid_action_mask(combat_view)], dtype=torch.bool, device=self._device
        )

        with torch.no_grad():
            probs, _ = self._model(
                *combat_view_encoding.as_tuple(), x_valid_action_mask=valid_action_mask_tensor
            )

        if self._greedy:
            action_idx = torch.argmax(probs)
        else:
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()

        action = action_idx_to_action(action_idx.item(), combat_view)

        return action, {"probs": probs}
