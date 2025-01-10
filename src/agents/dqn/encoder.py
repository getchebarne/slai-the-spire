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


def _encode_card_view(card_view: CardView, energy_current: int, card_count: int) -> list[int]:
    # playable, count, cost, damage, block, weak, discard, draw
    _list = [card_view.cost <= energy_current, card_count, card_view.cost, 0, 0, 0, 0, 0]
    for effect in card_view.effects:
        if effect.type in EFFECT_TYPE_MAP:
            _list[EFFECT_TYPE_MAP[effect.type] + 3] = effect.value

    return _list


def _encode_card_pad() -> list[int]:
    # playable, count, cost, damage, block, weak, discard, draw
    cost = 5
    playable = False
    card_count = 0
    return [playable, card_count, cost, 0, 0, 0, 0, 0]


def _encode_hand_view(
    hand_view: list[CardView], energy_current: int, device: torch.device
) -> torch.Tensor:
    _cards = []
    _active_mask = [0] * MAX_HAND_SIZE

    # Count occurrences of each card type in the hand
    card_type_counts = {}
    for card_view in hand_view:
        card_type = card_view.name  # Assuming `card_type` uniquely identifies a card
        card_type_counts[card_type] = card_type_counts.get(card_type, 0) + 1

    for idx, card_view in enumerate(hand_view):
        card_count = card_type_counts[card_view.name]
        _cards.append(_encode_card_view(card_view, energy_current, card_count))

        if card_view.is_active:
            _active_mask[idx] = 1

    _cards += [_encode_card_pad()] * (MAX_HAND_SIZE - len(_cards))
    return (
        torch.flatten(torch.tensor(_cards, device=device, dtype=torch.long)),
        torch.tensor(_active_mask, device=device, dtype=torch.long),
    )


def _encode_pile_view(pile_view: set[CardView], device: torch.device) -> torch.Tensor:
    if pile_view:
        _list = [_encode_card_view(card_view, -int(1e9), -int(1e9))[2:] for card_view in pile_view]

        return torch.sum(torch.tensor(_list, device=device, dtype=torch.long), dim=0)

    return torch.tensor([0] * 1 * 6, device=device, dtype=torch.long)


def encode_combat_view(combat_view: CombatView, device: torch.device) -> torch.Tensor:
    tensor_hand, tensor_active_mask = _encode_hand_view(
        combat_view.hand, combat_view.energy.current, device
    )

    return torch.concat(
        [
            torch.tensor([len(combat_view.hand)], device=device, dtype=torch.long),
            tensor_active_mask,
            tensor_hand,
            _encode_energy_view(combat_view.energy, device),
            # _encode_pile_view(combat_view.draw_pile, device),
            # _encode_pile_view(combat_view.disc_pile, device),
            _encode_character_view(combat_view.character, device),
            _encode_monster_views(combat_view.monsters, device),
            _encode_state_view(combat_view.state, device),
        ],
    )
