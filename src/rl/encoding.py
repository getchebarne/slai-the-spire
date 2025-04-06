# TODO: this is very tightly coupled with the models, improve it

import numpy as np
import torch
import torch.nn.functional as F

from src.game.combat.constant import MAX_HAND_SIZE
from src.game.combat.effect import EffectType
from src.game.combat.view import ActorView
from src.game.combat.view import CardView
from src.game.combat.view import CharacterView
from src.game.combat.view import CombatView
from src.game.combat.view import EnergyView
from src.game.combat.view import IntentView
from src.game.combat.view import ModifierViewType
from src.game.combat.view import MonsterView


MODIFIER_VIEW_TYPES = [modifier_view_type for modifier_view_type in ModifierViewType]

# TODO: create some sort of variable where I can access the effects defined in cards
# instead of doing this
EFFECT_TYPES = [
    EffectType.DEAL_DAMAGE,
    EffectType.GAIN_BLOCK,
    EffectType.DISCARD,
    EffectType.GAIN_WEAK,
    EffectType.DRAW_CARD,
]


def _one_hot_encode_(value: int, max_value: int, device: torch.device) -> torch.Tensor:
    num_bins = max_value + 1
    bin_idx = min(value, max_value)
    return F.one_hot(torch.tensor(bin_idx, device=device), num_classes=num_bins).to(torch.float32)


# TODO: in the future I'll need to add max-energy, it's always 3 for now
def _encode_energy_view(energy_view: EnergyView, device: torch.device) -> torch.Tensor:
    return _one_hot_encode_(energy_view.current, 3, device)


def _encode_actor_modifiers(actor_view: ActorView) -> list[int]:
    # Initialize as empty
    modifier_encodings = [0] * len(MODIFIER_VIEW_TYPES)
    for i, modifier_view_type in enumerate(MODIFIER_VIEW_TYPES):
        if modifier_view_type in actor_view.modifiers:
            stacks_current = actor_view.modifiers[modifier_view_type]
            if stacks_current is None:
                modifier_encodings[i] = 1
            else:
                modifier_encodings[i] = np.log(stacks_current + 1)

    return modifier_encodings


def _encode_character_view(character_view: CharacterView, device: torch.device) -> torch.Tensor:
    return torch.cat(
        [
            _one_hot_encode_(character_view.health_current, 50, device),
            _one_hot_encode_(character_view.block_current, 16, device),
            _one_hot_encode_(
                character_view.health_current + character_view.block_current, 66, device
            ),
            torch.tensor(
                _encode_actor_modifiers(character_view), dtype=torch.float32, device=device
            ),
        ],
    )


# TODO: adapt to multiple monsters
def _encode_monster_view(monster_view: MonsterView, device: torch.device) -> torch.Tensor:
    return torch.cat(
        [
            _one_hot_encode_(monster_view.health_current, 46, device),
            _one_hot_encode_(monster_view.block_current, 9, device),
            _one_hot_encode_(monster_view.health_current + monster_view.block_current, 55, device),
            # Intent
            torch.tensor(
                [
                    # monster_view.intent.instances or 0,
                    np.log((monster_view.intent.damage or 0) + 1),
                    monster_view.intent.block,
                    monster_view.intent.buff,
                    # Modifiers
                    *_encode_actor_modifiers(monster_view),
                ],
                dtype=torch.float32,
                device=device,
            ),
        ],
    )


# TODO: in the future, I should also encode the target and selection types
def _encode_card_view(card_view: CardView) -> list[int]:
    card_encoded = [0] * len(EFFECT_TYPES)
    for effect in card_view.effects:
        card_encoded[EFFECT_TYPES.index(effect.type)] = np.log(effect.value + 1)

    return card_encoded + [np.log(card_view.cost + 1)]


def _encode_card_views(card_views: list[CardView], device: torch.device) -> torch.Tensor:
    return torch.flatten(
        torch.tensor(
            [_encode_card_view(card_view) for card_view in card_views]
            + [
                [float("inf")] * get_card_encoding_dim()
                for _ in range(MAX_HAND_SIZE - len(card_views))
            ],
            dtype=torch.float32,
            device=device,
        )
    )


def get_character_encoding_dim() -> torch.Size:
    character_view_dummy = CharacterView("Dummy", 0, 0, 0, {})
    return _encode_character_view(character_view_dummy, torch.device("cpu")).shape


def get_monster_encoding_dim() -> torch.Size:
    monster_view_dummy = MonsterView("Dummy", 0, 0, 0, {}, IntentView(None, None, False, False), 0)
    return _encode_monster_view(monster_view_dummy, torch.device("cpu")).shape


def get_card_encoding_dim() -> int:
    card_view_dummy = CardView("Dummy", [], 0, False, 0)
    return len(_encode_card_view(card_view_dummy))


def get_energy_encoding_dim() -> torch.Size:
    energy_view_dummy = EnergyView(0, 0)
    return _encode_energy_view(energy_view_dummy, torch.device("cpu")).shape


def encode_combat_view(
    combat_view: CombatView, device: torch.device
) -> tuple[torch.Tensor, dict[int, int]]:
    card_active_mask = [False] * MAX_HAND_SIZE
    for i, card_view in enumerate(combat_view.hand):
        if card_view.is_active:
            card_active_mask[i] = True
            break

    encoding = torch.cat(
        [
            # Hand size
            torch.tensor([len(combat_view.hand)], dtype=torch.float32),
            # Active card
            torch.tensor(card_active_mask, dtype=torch.float32),
            # Hand
            _encode_card_views(combat_view.hand, device),
            # Character
            _encode_character_view(combat_view.character, device),
            # Monster
            _encode_monster_view(combat_view.monsters[0], device),
            # Energy
            _encode_energy_view(combat_view.energy, device),
        ],
    )

    return encoding
