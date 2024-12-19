import math

import torch
import numpy as np

from src.game.combat.constant import MAX_HAND_SIZE
from src.game.combat.entities import EffectType
from src.game.combat.view import CombatView
from src.game.combat.view import StateView
from src.game.combat.view.actor import ModifierViewType
from src.game.combat.view.card import CardView
from src.game.combat.view.character import CharacterView
from src.game.combat.view.energy import EnergyView
from src.game.combat.view.monster import MonsterView


MODIFIER_TYPES = [ModifierViewType.WEAK, ModifierViewType.STR]
EFFECT_TYPE_MAP = {
    EffectType.DEAL_DAMAGE: 0,
    EffectType.GAIN_BLOCK: 1,
    EffectType.DISCARD: 2,
    EffectType.GAIN_WEAK: 3,
    EffectType.DRAW_CARD: 4,
}
MAX_LEN_DISC_PILE = 12
MAX_LEN_DRAW_PILE = 9


def _encode_state_view(state_view: StateView, device: torch.device) -> torch.Tensor:
    if state_view == StateView.DEFAULT:
        return torch.tensor([1, 0, 0], device=device, dtype=torch.float32)

    if state_view == StateView.AWAIT_CARD_TARGET:
        return torch.tensor([0, 1, 0], device=device, dtype=torch.float32)

    if state_view == StateView.AWAIT_EFFECT_TARGET:
        return torch.tensor([0, 0, 1], device=device, dtype=torch.float32)

    raise ValueError(f"Unsupported state_view: {state_view}")


def _encode_energy_view(energy_view: EnergyView, device: torch.device) -> torch.Tensor:
    return torch.tensor([np.log(energy_view.current + 1)], device=device, dtype=torch.float32)


def _encode_character_view(character_view: CharacterView, device: torch.device) -> torch.Tensor:
    # Modifiers
    modifier_stacks = [
        next(
            (
                np.log(modifier.stacks + 1)
                for modifier in character_view.modifiers
                if modifier.type == modifier_type
            ),
            0,
        )
        for modifier_type in MODIFIER_TYPES
    ]

    return torch.tensor(
        [
            np.log(character_view.health.current + 1),
            np.log(character_view.block.current + 1),
            np.log(character_view.health.current + character_view.block.current + 1),
            *modifier_stacks,
        ],
        device=device,
        dtype=torch.float32,
    )


# TODO: add support for multiple monsters
def _encode_monster_views(monster_views: list[MonsterView], device: torch.device) -> torch.Tensor:
    _tensors = []
    for monster_view in monster_views:
        # Modifiers
        modifier_stacks = [
            next(
                (
                    np.log(modifier.stacks + 1)
                    for modifier in monster_view.modifiers
                    if modifier.type == modifier_type
                ),
                0,
            )
            for modifier_type in MODIFIER_TYPES
        ]
        _tensors.append(
            torch.tensor(
                [
                    np.log(monster_view.health.current + 1),
                    np.log(monster_view.block.current + 1),
                    np.log(monster_view.health.current + monster_view.block.current + 1),
                    # Intent
                    np.log(monster_view.intent.damage or 0 + 1),
                    np.log(monster_view.intent.instances or 0 + 1),
                    monster_view.intent.block,
                    monster_view.intent.buff,
                    # Modifiers
                    *modifier_stacks,
                ],
                device=device,
                dtype=torch.float32,
            )
        )

    return torch.concat(_tensors)


def _encode_card_view(card_view: CardView, energy_current: int) -> list[int]:
    # cost, damage, block, weak, discard, draw,
    _list = [np.log(card_view.cost + 1), 0, 0, 0, 0, 0, np.log(energy_current + 1)]
    for effect in card_view.effects:
        if effect.type in EFFECT_TYPE_MAP:
            _list[EFFECT_TYPE_MAP[effect.type] + 1] = np.log(effect.value + 1)

    return _list


def _encode_card_pad(energy_current: int) -> list[int]:
    # cost, damage, block, weak, discard, draw
    return [np.log(10 + 1), 0, 0, 0, 0, 0, np.log(energy_current + 1)]


def _encode_hand_view(
    hand_view: list[CardView], energy_current: int, device: torch.device
) -> torch.Tensor:
    _list = []
    _mask = [0] * MAX_HAND_SIZE
    for idx, card_view in enumerate(hand_view):
        _list.append(_encode_card_view(card_view, energy_current))

        if card_view.is_active:
            _mask[idx] = 1

    _list += [_encode_card_pad(energy_current)] * (MAX_HAND_SIZE - len(hand_view))

    return torch.flatten(torch.tensor(_list, device=device, dtype=torch.float32)), torch.tensor(
        _mask, device=device, dtype=torch.float32
    )


def _encode_pile_view(pile_view: set[CardView], device: torch.device) -> torch.Tensor:
    if pile_view:
        _list = [_encode_card_view(card_view, math.inf)[:-1] for card_view in pile_view]

        return torch.cat(
            [
                torch.sum(torch.tensor(_list, device=device, dtype=torch.float32), dim=0),
                torch.mean(torch.tensor(_list, device=device, dtype=torch.float32), dim=0),
                torch.max(torch.tensor(_list, device=device, dtype=torch.float32), dim=0)[0],
            ]
        )

    return torch.tensor([0] * 3 * 6, device=device, dtype=torch.float32)


def encode_combat_view(combat_view: CombatView, device: torch.device) -> torch.Tensor:
    tensor_hand, tensor_mask = _encode_hand_view(
        combat_view.hand, combat_view.energy.current, device
    )
    # print(tensor_mask.shape)
    # print(tensor_hand.shape)
    # print(_encode_pile_view(combat_view.draw_pile, device).shape)
    # print(_encode_pile_view(combat_view.disc_pile, device).shape)
    # print("#" * 10)
    return torch.concat(
        [
            tensor_mask,
            tensor_hand,
            _encode_pile_view(combat_view.draw_pile, device),
            _encode_pile_view(combat_view.disc_pile, device),
            _encode_energy_view(combat_view.energy, device),
            _encode_character_view(combat_view.character, device),
            _encode_monster_views(combat_view.monsters, device),
            _encode_state_view(combat_view.state, device),
        ],
    )
