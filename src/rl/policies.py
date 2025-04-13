import random
from abc import ABC, abstractmethod
from typing import Any, TypeAlias

import torch
import torch.nn as nn

from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.view import CombatView
from src.rl.encoding import encode_combat_view
from src.rl.models.interface import action_idx_to_action
from src.rl.models.interface import get_valid_action_mask


SelectActionMetadata: TypeAlias = dict[str, Any]


class PolicyBase(ABC):
    @abstractmethod
    def select_action(combat_view: CombatView) -> tuple[Action, SelectActionMetadata]:
        raise NotImplementedError


class PolicyRandom(PolicyBase):
    def select_action(self, combat_view: CombatView) -> tuple[Action, SelectActionMetadata]:
        if any([card.is_active for card in combat_view.hand]):
            return Action(ActionType.SELECT_ENTITY, combat_view.monsters[0].entity_id), {}

        if combat_view.effect is not None:
            return (
                Action(
                    ActionType.SELECT_ENTITY,
                    random.choice([card.entity_id for card in combat_view.hand]),
                ),
                {},
            )

        id_selectable_cards = [
            card.entity_id for card in combat_view.hand if card.cost <= combat_view.energy.current
        ]
        if id_selectable_cards:
            return Action(ActionType.SELECT_ENTITY, random.choice(id_selectable_cards)), {}

        return Action(ActionType.END_TURN), {}


class PolicyQMax(PolicyBase):
    def __init__(self, model: nn.Module, device: torch.device):
        self._model = model
        self._device = device

        # Send model to device
        self._model.to(self._device)

    def select_action(self, combat_view: CombatView) -> tuple[Action, SelectActionMetadata]:
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

    def select_action(self, combat_view: CombatView) -> tuple[Action, SelectActionMetadata]:
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
