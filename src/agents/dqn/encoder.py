import torch

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
COST_PAD = 5
EFFECT_TYPE_MAP = {
    EffectType.DEAL_DAMAGE: 0,
    EffectType.GAIN_BLOCK: 1,
    EffectType.DISCARD: 2,
    EffectType.GAIN_WEAK: 3,
    EffectType.DRAW_CARD: 4,
}


def _encode_state_view(state_view: StateView, device: torch.device) -> torch.Tensor:
    if state_view == StateView.DEFAULT:
        return torch.tensor([1, 0, 0], device=device, dtype=torch.long)

    if state_view == StateView.AWAIT_CARD_TARGET:
        return torch.tensor([0, 1, 0], device=device, dtype=torch.long)

    if state_view == StateView.AWAIT_EFFECT_TARGET:
        return torch.tensor([0, 0, 1], device=device, dtype=torch.long)

    raise ValueError(f"Unsupported state_view: {state_view}")


def _encode_energy_view(energy_view: EnergyView, device: torch.device) -> torch.Tensor:
    return torch.tensor([energy_view.current], device=device, dtype=torch.long)


def _encode_character_view(character_view: CharacterView, device: torch.device) -> torch.Tensor:
    # Modifiers
    modifier_stacks = [
        next(
            (
                modifier.stacks
                for modifier in character_view.modifiers
                if modifier.type == modifier_type
            ),
            0,
        )
        for modifier_type in MODIFIER_TYPES
    ]

    return torch.tensor(
        [
            character_view.health.current,
            character_view.block.current,
            character_view.health.current + character_view.block.current,
            *modifier_stacks,
        ],
        device=device,
        dtype=torch.long,
    )


# TODO: add support for multiple monsters
def _encode_monster_views(monster_views: list[MonsterView], device: torch.device) -> torch.Tensor:
    _tensors = []
    for monster_view in monster_views:
        # Modifiers
        modifier_stacks = [
            next(
                (
                    modifier.stacks
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
                    monster_view.health.current,
                    monster_view.block.current,
                    monster_view.health.current + monster_view.block.current,
                    # Intent
                    monster_view.intent.damage or 0,
                    monster_view.intent.instances or 0,
                    monster_view.intent.block,
                    monster_view.intent.buff,
                    # Modifiers
                    *modifier_stacks,
                ],
                device=device,
                dtype=torch.long,
            )
        )

    return torch.concat(_tensors)


def _encode_card_view(card_view: CardView) -> list[int]:
    # cost, damage, block, weak, discard, draw,
    _list = [card_view.cost, 0, 0, 0, 0, 0]
    for effect in card_view.effects:
        if effect.type in EFFECT_TYPE_MAP:
            _list[EFFECT_TYPE_MAP[effect.type] + 1] = effect.value

    return _list


def _encode_card_pad() -> list[int]:
    # cost, damage, block, weak, discard, draw
    return [COST_PAD, 0, 0, 0, 0, 0]


def _encode_hand_view(hand_view: list[CardView], device: torch.device) -> torch.Tensor:
    _cards = []
    _costs = []
    _mask = [0] * MAX_HAND_SIZE
    for idx, card_view in enumerate(hand_view):
        cost_card = _encode_card_view(card_view)
        cost = cost_card[0]
        card = cost_card[1:]
        _costs.append(cost)
        _cards.append(card)

        if card_view.is_active:
            _mask[idx] = 1

    for i in range(MAX_HAND_SIZE - len(hand_view)):
        cost_card = _encode_card_pad()
        cost = cost_card[0]
        card = cost_card[1:]
        _costs.append(cost)
        _cards.append(card)

    return (
        torch.tensor(_costs, device=device, dtype=torch.long),
        torch.flatten(torch.tensor(_cards, device=device, dtype=torch.long)),
        torch.tensor(_mask, device=device, dtype=torch.long),
    )


def _encode_pile_view(pile_view: set[CardView], device: torch.device) -> torch.Tensor:
    if pile_view:
        _list = [_encode_card_view(card_view) for card_view in pile_view]

        return torch.sum(torch.tensor(_list, device=device, dtype=torch.long), dim=0)

    return torch.tensor([0] * 1 * 6, device=device, dtype=torch.long)


def encode_combat_view(combat_view: CombatView, device: torch.device) -> torch.Tensor:
    tensor_costs, tensor_cards, tensor_mask = _encode_hand_view(combat_view.hand, device)

    return torch.concat(
        [
            torch.tensor([len(combat_view.hand)], device=device, dtype=torch.long),
            tensor_mask,
            tensor_costs,
            tensor_cards,
            _encode_energy_view(combat_view.energy, device),
            _encode_pile_view(combat_view.draw_pile, device),
            _encode_pile_view(combat_view.disc_pile, device),
            _encode_character_view(combat_view.character, device),
            _encode_monster_views(combat_view.monsters, device),
            _encode_state_view(combat_view.state, device),
        ],
    )
