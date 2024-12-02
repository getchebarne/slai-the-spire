from typing import Optional

import torch
import torch.nn as nn

from src.game.combat.constant import MAX_HAND_SIZE
from src.game.combat.constant import MAX_MONSTERS
from src.game.combat.entities import CardName
from src.game.combat.view import CombatView
from src.game.combat.view.card import CardView
from src.game.combat.view.character import CharacterView
from src.game.combat.view.energy import EnergyView
from src.game.combat.view.monster import MonsterView
from src.game.combat.view.state import StateView


def _encode_state(state_view: StateView, embedding_table_state: nn.Embedding) -> torch.Tensor:
    # TODO: move this
    if state_view == StateView.DEFAULT:
        state_idx = 0

    elif state_view == StateView.AWAIT_CARD_TARGET:
        state_idx = 1

    elif state_view == StateView.AWAIT_EFFECT_TARGET:
        state_idx = 2

    return embedding_table_state(torch.tensor(state_idx))


def _encode_character(character_view: CharacterView) -> torch.Tensor:
    return torch.tensor(
        [character_view.health.max, character_view.health.current, character_view.block.current]
    )


def _encode_energy(energy_view: EnergyView) -> torch.Tensor:
    return torch.tensor([energy_view.max, energy_view.current])


def _encode_monsters(monster_views: list[MonsterView]) -> torch.Tensor:
    return torch.concat(
        [
            torch.tensor(
                [
                    monster_view.health.max,
                    monster_view.health.current,
                    monster_view.block.current,
                    0 if monster_view.intent.damage is None else monster_view.intent.damage[0],
                    0 if monster_view.intent.damage is None else monster_view.intent.damage[1],
                    int(monster_view.intent.block),
                ]
            )
            for monster_view in monster_views
        ],
    )


def _encode_card(
    card_view: Optional[CardView], embbeding_table_card_name: nn.Embedding
) -> torch.Tensor:
    if card_view is None:
        # TODO: move this
        PAD_COST = -1
        PAD_IS_ACTIVE = False
        PAD_INDEX = 0
        return torch.concat(
            [
                torch.tensor((PAD_COST, PAD_IS_ACTIVE)),
                embbeding_table_card_name(torch.tensor(PAD_INDEX)),
            ]
        )

    # TODO: move this
    if card_view.name == CardName.STRIKE:
        card_name_idx = 1

    elif card_view.name == CardName.DEFEND:
        card_name_idx = 2

    elif card_view.name == CardName.NEUTRALIZE:
        card_name_idx = 3

    elif card_view.name == CardName.SURVIVOR:
        card_name_idx = 4

    return torch.concat(
        [
            torch.tensor((card_view.cost, card_view.is_active)),
            embbeding_table_card_name(torch.tensor(card_name_idx)),
        ]
    )


def _encode_hand(hand: list[CardView], embbeding_table_card_name: nn.Embedding) -> torch.Tensor:
    # Pad to MAX_HAND_SIZE cards
    hand_pad = hand + [None] * (MAX_HAND_SIZE - len(hand))

    return torch.flatten(
        torch.concat([_encode_card(card, embbeding_table_card_name) for card in hand_pad])
    )


def _encode_combat(
    combat_view: CombatView,
    embbeding_table_card_name: nn.Embedding,
    embedding_table_state: nn.Embedding,
) -> torch.Tensor:
    return torch.flatten(
        torch.concat(
            [
                _encode_state(combat_view.state, embedding_table_state),
                _encode_character(combat_view.character),
                _encode_monsters(combat_view.monsters),
                _encode_hand(combat_view.hand, embbeding_table_card_name),
                _encode_energy(combat_view.energy),
            ]
        )
    )


class MLP(nn.Module):
    def __init__(self, layer_sizes: list[int], bias: bool = True, activation_name: str = "ReLU"):
        super().__init__()

        self._layer_sizes = layer_sizes
        self._bias = bias
        self._activation_name = activation_name

        self._mlp = self._create_mlp()

    def _create_mlp(self) -> nn.Module:
        layers = []
        for i in range(len(self._layer_sizes)):
            if i == 0:
                # First layer is lazy
                layers.append(nn.LazyLinear(self._layer_sizes[i], self._bias))
            else:
                layers.append(
                    nn.Linear(self._layer_sizes[i - 1], self._layer_sizes[i], self._bias)
                )

            layers.append(getattr(nn, self._activation_name)())

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._mlp(x)


class EmbeddingMLP(nn.Module):
    def __init__(
        self, card_name_embedding_size: int, state_embedding_size: int, linear_sizes: list[int]
    ):
        super().__init__()
        self._card_name_embedding_size = card_name_embedding_size
        self._state_embedding_size = state_embedding_size
        self._linear_sizes = linear_sizes

        self._embedding_table_card_name = nn.Embedding(
            len(CardName) + 1, card_name_embedding_size, padding_idx=0
        )
        self._embedding_table_state = nn.Embedding(len(StateView), state_embedding_size)
        self._mlp = MLP(linear_sizes)
        self._last = nn.Linear(linear_sizes[-1], MAX_HAND_SIZE + MAX_MONSTERS + 1, bias=True)

    def forward(self, combat_views: list[CombatView]) -> torch.Tensor:
        x = torch.stack(
            [
                _encode_combat(
                    combat_view, self._embedding_table_card_name, self._embedding_table_state
                )
                for combat_view in combat_views
            ]
        )

        x = self._mlp(x)
        x = self._last(x)

        return x
