import torch

from src.game.combat.constant import MAX_HAND_SIZE
from src.game.combat.effect import EffectType
from src.game.combat.view import CombatView
from src.game.combat.view.card import CardView
from src.game.combat.view.character import CharacterView
from src.game.combat.view.energy import EnergyView
from src.game.combat.view.monster import MonsterView


# TODO: add more effects
EFFECT_TYPE_MAP = {
    EffectType.DEAL_DAMAGE: 0,
    EffectType.GAIN_BLOCK: 1,
    EffectType.DISCARD: 2,
    EffectType.GAIN_WEAK: 3,
}


def _encode_energy_view(energy_view: EnergyView, device: torch.device) -> torch.Tensor:
    return torch.tensor([energy_view.current], device=device, dtype=torch.float32)


# TODO: add modifiers
def _encode_character_view(character_view: CharacterView, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            character_view.health_current,
            character_view.block_current,
            character_view.health_current + character_view.block_current,
        ],
        dtype=torch.float32,
        device=device,
    )


# TODO: add support for multiple monsters
# TODO: add modifiers
def _encode_monster_views(monster_views: list[MonsterView], device: torch.device) -> torch.Tensor:
    tensors = []
    for monster_view in monster_views:
        tensors.append(
            torch.tensor(
                [
                    monster_view.health_current,
                    monster_view.block_current,
                    monster_view.health_current + monster_view.block_current,
                    # Intent
                    monster_view.intent.damage or 0,
                    monster_view.intent.instances or 0,
                    monster_view.intent.block,
                    # monster_view.intent.buff,
                    monster_view.modifier_weak.stacks_current,
                ],
                dtype=torch.float32,
                device=device,
            )
        )

    return torch.cat(tensors)


def _encode_card_view(card_view: CardView, energy_current: int) -> list[int]:
    # damage, block, discard, gain_weak, playable, cost, is_active
    card_encoded = [0, 0, 0, 0]
    for effect in card_view.effects:
        card_encoded[EFFECT_TYPE_MAP[effect.type]] = effect.value

    return card_encoded + [card_view.cost <= energy_current, card_view.cost, card_view.is_active]


def _encode_card_view_pad() -> list[int]:
    # damage, block, discard, gain_weak, playable, cost, is_active
    cost = 5
    playable = False
    is_active = False
    return [0, 0, 0, 0, playable, cost, is_active]


def _encode_hand_view(
    hand_view: list[CardView], energy_current: int, device: torch.device
) -> torch.Tensor:
    card_views_encoded = []

    for card_view in hand_view:
        card_views_encoded.append(_encode_card_view(card_view, energy_current))

    # Padded cards
    card_views_encoded += [_encode_card_view_pad()] * (MAX_HAND_SIZE - len(hand_view))

    return torch.flatten(torch.tensor(card_views_encoded, dtype=torch.float32, device=device))


def encode_combat_view(combat_view: CombatView, device: torch.device) -> torch.Tensor:
    return torch.cat(
        [
            _encode_hand_view(combat_view.hand, combat_view.energy.current, device),
            _encode_energy_view(combat_view.energy, device),
            _encode_character_view(combat_view.character, device),
            _encode_monster_views(combat_view.monsters, device),
        ],
    )
