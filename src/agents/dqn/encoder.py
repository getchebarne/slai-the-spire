import torch

from src.game.combat.constant import MAX_HAND_SIZE
from src.game.combat.entities import EffectType
from src.game.combat.view import CombatView
from src.game.combat.view.actor import ModifierViewType
from src.game.combat.view.card import CardView
from src.game.combat.view.character import CharacterView
from src.game.combat.view.energy import EnergyView
from src.game.combat.view.monster import MonsterView
from src.game.combat.view.state import StateView


MODIFIER_TYPES = [ModifierViewType.WEAK, ModifierViewType.STR]
MAX_LEN_DISC_PILE = 10
MAX_LEN_DRAW_PILE = 7


def _encode_state_view(state_view: StateView, device: torch.device) -> torch.Tensor:
    if state_view == StateView.DEFAULT:
        return torch.tensor([1, 0, 0], device=device, dtype=torch.float32)

    if state_view == StateView.AWAIT_CARD_TARGET:
        return torch.tensor([0, 1, 0], device=device, dtype=torch.float32)

    if state_view == StateView.AWAIT_EFFECT_TARGET:
        return torch.tensor([0, 0, 1], device=device, dtype=torch.float32)

    raise ValueError(f"Unsupported state_view: {state_view}")


def _encode_energy_view(energy_view: EnergyView, device: torch.device) -> torch.Tensor:
    return torch.tensor([energy_view.current], device=device, dtype=torch.float32)


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
        [character_view.health.current, character_view.block.current, *modifier_stacks],
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
                    # Intent
                    monster_view.intent.damage or 0,
                    monster_view.intent.instances or 0,
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


def _encode_card_view(card_view: CardView, device: torch.device) -> torch.Tensor:
    effect_type_map = {
        EffectType.DEAL_DAMAGE: 0,
        EffectType.GAIN_BLOCK: 1,
        EffectType.GAIN_WEAK: 2,
        EffectType.DISCARD: 3,
    }
    # damage, block, weak, discard, cost, is_active
    _list = [0, 0, 0, 0, card_view.cost, card_view.is_active]
    for effect in card_view.effects:
        if effect.type in effect_type_map:
            _list[effect_type_map[effect.type]] = effect.value

    return torch.tensor(_list, device=device, dtype=torch.float32)


def _encode_hand_view(hand: list[CardView], device: torch.device) -> torch.Tensor:
    # Initialize tensors to store CardName indexes and metadata (is_active and cost)
    _tensor = torch.zeros((MAX_HAND_SIZE + 1, 6), device=device, dtype=torch.float32)

    # Iterate over batch samples
    for idx, card_view in enumerate(hand):
        # Card name
        _tensor[idx] = _encode_card_view(card_view, device)

    _tensor[-1] = torch.sum(torch.sum(_tensor[:MAX_HAND_SIZE], dim=0))

    return torch.flatten(_tensor)


def _encode_pile_view(pile: set[CardView], max_len: int, device: torch.device) -> torch.Tensor:
    # Pre-allocate tensor with zeros for padding
    _tensor = torch.zeros((max_len + 1, 6), device=device, dtype=torch.float32)

    # Sort pile to ensure consistent encoding
    pile_sorted = sorted(pile, key=lambda x: x.name.name)

    # Iterate
    for idx, card_view in enumerate(pile_sorted):
        _tensor[idx] = _encode_card_view(card_view, device)

    _tensor[-1] = torch.sum(_tensor[:max_len], dim=0)

    return torch.flatten(_tensor)


def encode_combat_view(combat_view: CombatView, device: torch.device) -> torch.Tensor:
    return torch.concat(
        [
            _encode_state_view(combat_view.state, device),
            _encode_energy_view(combat_view.energy, device),
            _encode_character_view(combat_view.character, device),
            _encode_monster_views(combat_view.monsters, device),
            _encode_hand_view(combat_view.hand, device),
            _encode_pile_view(combat_view.draw_pile, MAX_LEN_DRAW_PILE, device),
            _encode_pile_view(combat_view.disc_pile, MAX_LEN_DISC_PILE, device),
        ],
    )
