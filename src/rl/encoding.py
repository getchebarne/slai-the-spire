from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.game.combat.constant import EFFECT_TYPE_CARD
from src.game.combat.constant import MAX_SIZE_DISC_PILE
from src.game.combat.constant import MAX_SIZE_DRAW_PILE
from src.game.combat.constant import MAX_SIZE_HAND
from src.game.combat.view import CardView
from src.game.combat.view import CharacterView
from src.game.combat.view import CombatView
from src.game.combat.view import EffectType
from src.game.combat.view import EnergyView
from src.game.combat.view import IntentView
from src.game.combat.view import ModifierViewType
from src.game.combat.view import MonsterView


# TODO: find better place for these constants
MODIFIER_VIEW_TYPES = [modifier_view_type for modifier_view_type in ModifierViewType]
EFFECT_TYPE_CARD_POS = {
    effect_type_card: idx for idx, effect_type_card in enumerate(EFFECT_TYPE_CARD)
}
EFFECT_TYPES_MAX = {
    EffectType.DEAL_DAMAGE: 10,
    EffectType.GAIN_BLOCK: 11,
    EffectType.DISCARD: 1,
    EffectType.GAIN_WEAK: 1,
    EffectType.DRAW_CARD: 2,
}
MODIFIER_VIEW_TYPES_MAX = {
    ModifierViewType.STRENGTH: 20,
    ModifierViewType.WEAK: 4,
}
COST_MAX = 2
DAMAGE_MAX = 32


@dataclass(frozen=True)
class CombatViewEncoding:
    x_len_hand: torch.Tensor
    x_len_draw: torch.Tensor
    x_len_disc: torch.Tensor
    x_card_active_mask: torch.Tensor
    x_card_hand: torch.Tensor
    x_card_draw: torch.Tensor
    x_card_disc: torch.Tensor
    x_char: torch.Tensor
    x_monster: torch.Tensor
    x_energy: torch.Tensor

    def as_tuple(self) -> tuple[torch.Tensor, ...]:
        return (
            self.x_len_hand,
            self.x_len_draw,
            self.x_len_disc,
            self.x_card_active_mask,
            self.x_card_hand,
            self.x_card_draw,
            self.x_card_disc,
            self.x_char,
            self.x_monster,
            self.x_energy,
        )


def _one_hot_encode(
    value: int, value_min: int, value_max: int, device: torch.device
) -> torch.Tensor:
    value_clamp = max(min(value, value_max), value_min)
    bin_idx = value_clamp - value_min
    num_bins = value_max - value_min + 1
    return F.one_hot(torch.tensor(bin_idx, device=device), num_classes=num_bins).to(torch.float32)


# TODO: in the future I'll need to add max-energy, it's always 3 for now
def _encode_energy_view(energy_view: EnergyView, device: torch.device) -> torch.Tensor:
    return _one_hot_encode(energy_view.current, value_min=0, value_max=3, device=device)


def _encode_actor_view_modifiers(
    actor_view_modifiers: dict[ModifierViewType, int | None], device: torch.device
) -> torch.Tensor:
    # Initialize as zeros
    modifier_encodings = [0] * len(MODIFIER_VIEW_TYPES)
    for i, modifier_view_type in enumerate(MODIFIER_VIEW_TYPES):
        if modifier_view_type in actor_view_modifiers:
            stacks_current = actor_view_modifiers[modifier_view_type]
            if stacks_current is None:
                modifier_encodings[i] = 1
            else:
                modifier_encodings[i] = (
                    stacks_current / MODIFIER_VIEW_TYPES_MAX[modifier_view_type]
                )

    return torch.tensor(modifier_encodings, dtype=torch.float32, device=device)


def _encode_char_view(
    char_view: CharacterView,
    health_min: int,
    health_max: int,
    block_max: int,
    device: torch.device,
) -> torch.Tensor:
    block_min = 0
    total_min = health_min + block_min
    total_max = health_max + block_max

    return torch.cat(
        [
            _one_hot_encode(char_view.health_current, health_min, health_max, device),
            _one_hot_encode(char_view.block_current, block_min, block_max, device),
            _one_hot_encode(
                char_view.health_current + char_view.block_current, total_min, total_max, device
            ),
            _encode_actor_view_modifiers(char_view.modifiers, device),
        ],
    )


def _encode_intent_view(intent_view: IntentView, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            (intent_view.damage or 0) / DAMAGE_MAX,
            intent_view.block,
            intent_view.buff,
            # monster_view.intent.instances or 0,
        ],
        dtype=torch.float32,
        device=device,
    )


# TODO: adapt to multiple monsters
def _encode_monster_view(
    monster_view: MonsterView,
    health_min: int,
    health_max: int,
    block_max: int,
    device: torch.device,
) -> torch.Tensor:
    block_min = 0
    total_min = health_min + block_min
    total_max = health_max + block_max

    return torch.cat(
        [
            _one_hot_encode(monster_view.health_current, health_min, health_max, device),
            _one_hot_encode(monster_view.block_current, block_min, block_max, device),
            _one_hot_encode(
                monster_view.health_current + monster_view.block_current,
                total_min,
                total_max,
                device,
            ),
            _encode_actor_view_modifiers(monster_view.modifiers, device),
            # Intent
            _encode_intent_view(monster_view.intent, device),
        ],
    )


# TODO: in the future, I should also encode the target and selection types
def _encode_card_view(card_view: CardView, device: torch.device) -> torch.Tensor:
    card_encoded = [0] * len(EFFECT_TYPE_CARD_POS) + [card_view.cost / COST_MAX]
    for effect in card_view.effects:
        card_encoded[EFFECT_TYPE_CARD_POS[effect.type]] = (
            effect.value / EFFECT_TYPES_MAX[effect.type]
        )

    return torch.tensor(card_encoded, dtype=torch.float32, device=device)


def _encode_card_views(
    card_views: list[CardView], max_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    card_views_len = len(card_views)
    if card_views_len > max_size:
        raise ValueError(
            f"Length of card views ({card_views_len}) exceeds maximum length ({max_size})"
        )

    mask = torch.zeros(max_size, dtype=torch.float32, device=device)
    if not card_views:
        # Return all-zeros tensor of shape (`max_size`, `card_encoding_dim`) and zero-mask
        card_encoding_dim = get_card_encoding_dim()
        return torch.zeros(max_size, card_encoding_dim, dtype=torch.float32, device=device), mask

    card_view_encodings = None
    for idx, card_view in enumerate(card_views):
        # Get encoding
        card_view_encoding = _encode_card_view(card_view, device)

        if card_view_encodings is None:
            # Intialize all-zeros tensor to hold all encodings, now that we now the enc. dimension
            card_view_encodings = torch.zeros(max_size, card_view_encoding.shape[0])

        # Assign encoding
        card_view_encodings[idx] = card_view_encoding

        # Set mask index
        if card_view.is_active:
            mask[idx] = 1.0

    return card_view_encodings, mask


def get_character_encoding_dim() -> int:
    character_view_dummy = CharacterView("Dummy", 0, 0, 0, {})
    return _encode_char_view(character_view_dummy, 1, 50, 16, torch.device("cpu")).shape[0]


def get_monster_encoding_dim() -> int:
    monster_view_dummy = MonsterView("Dummy", 0, 0, 0, {}, IntentView(None, None, False, False), 0)
    return _encode_monster_view(monster_view_dummy, 1, 46, 9, torch.device("cpu")).shape[0]


def get_card_encoding_dim() -> int:
    card_view_dummy = CardView("Dummy", [], 0, False, 0)
    return _encode_card_view(card_view_dummy, torch.device("cpu")).shape[0]


def get_energy_encoding_dim() -> int:
    energy_view_dummy = EnergyView(0, 0)
    return _encode_energy_view(energy_view_dummy, torch.device("cpu")).shape[0]


def encode_combat_view(combat_view: CombatView, device: torch.device) -> tuple[torch.Tensor, ...]:
    # Get number of cards in each collection
    x_len_hand = torch.tensor([len(combat_view.hand)], dtype=torch.float32, device=device)
    x_len_draw = torch.tensor([len(combat_view.draw_pile)], dtype=torch.float32, device=device)
    x_len_disc = torch.tensor([len(combat_view.disc_pile)], dtype=torch.float32, device=device)

    # Encode cards in each collection
    x_card_hand, x_card_active_mask = _encode_card_views(combat_view.hand, MAX_SIZE_HAND, device)
    x_card_draw, _ = _encode_card_views(combat_view.draw_pile, MAX_SIZE_DRAW_PILE, device)
    x_card_disc, _ = _encode_card_views(combat_view.disc_pile, MAX_SIZE_DISC_PILE, device)

    # Encode character
    x_char = _encode_char_view(
        combat_view.character,
        health_min=1,
        health_max=50,
        block_max=16,  # Leg Sweep + Defend (or Backflip)
        device=device,
    )

    # Encode monster
    x_monster = _encode_monster_view(
        combat_view.monsters[0],
        health_min=1,
        health_max=46,  # TODO: will change
        block_max=9,  # Bellow
        device=device,
    )

    # Encode energy
    x_energy = _encode_energy_view(combat_view.energy, device)

    return CombatViewEncoding(
        x_len_hand.view(1, 1),
        x_len_draw.view(1, 1),
        x_len_disc.view(1, 1),
        x_card_active_mask.view(1, MAX_SIZE_HAND),
        x_card_hand.view(1, MAX_SIZE_HAND, -1),
        x_card_draw.view(1, MAX_SIZE_DRAW_PILE, -1),
        x_card_disc.view(1, MAX_SIZE_DISC_PILE, -1),
        x_char.view(1, -1),
        x_monster.view(1, -1),
        x_energy.view(1, -1),
    )


def pack_combat_view_encoding(combat_view_encoding: CombatViewEncoding) -> torch.Tensor:
    x_all = torch.cat(
        [
            combat_view_encoding.x_len_hand,
            combat_view_encoding.x_len_draw,
            combat_view_encoding.x_len_disc,
            combat_view_encoding.x_card_active_mask,
            torch.flatten(combat_view_encoding.x_card_hand, start_dim=1),
            torch.flatten(combat_view_encoding.x_card_draw, start_dim=1),
            torch.flatten(combat_view_encoding.x_card_disc, start_dim=1),
            combat_view_encoding.x_char,
            combat_view_encoding.x_monster,
            combat_view_encoding.x_energy,
        ],
        dim=1,
    )

    return x_all.view(1, -1)  # Add batch dimension


def unpack_combat_view_encoding(
    x: torch.Tensor,
    dim_enc_card: int = get_card_encoding_dim(),
    dim_enc_char: int = get_character_encoding_dim(),
    dim_enc_monster: int = get_monster_encoding_dim(),
    dim_enc_energy: int = get_energy_encoding_dim(),
) -> CombatViewEncoding:
    # Input shape: (batch_size, D), where:
    # D = (
    #   3 +
    #   `MAX_SIZE_HAND` +
    #   `MAX_SIZE_HAND` x `dim_enc_card`` +
    #   `MAX_SIZE_DRAW_PILE` x `dim_enc_card` +
    #   `MAX_SIZE_DISC_PILE` x `dim_enc_card` +
    #   `dim_enc_char` +
    #   `dim_enc_monster` +
    #   `dim_enc_energy`
    # )
    batch_size = x.shape[0]
    idx = 0

    # Card collection lengths
    x_len_hand = x[:, idx : idx + 1]
    idx += 1
    x_len_draw = x[:, idx : idx + 1]
    idx += 1
    x_len_disc = x[:, idx : idx + 1]
    idx += 1

    # Active card mask
    x_card_active_mask = x[:, idx : idx + MAX_SIZE_HAND]
    idx += MAX_SIZE_HAND

    # Hand
    sz_hand = MAX_SIZE_HAND * dim_enc_card
    x_card_hand = x[:, idx : idx + sz_hand].view(batch_size, MAX_SIZE_HAND, dim_enc_card)
    idx += sz_hand

    # Draw pile
    sz_draw = MAX_SIZE_DRAW_PILE * dim_enc_card
    x_card_draw = x[:, idx : idx + sz_draw].view(batch_size, MAX_SIZE_DRAW_PILE, dim_enc_card)
    idx += sz_draw

    # Discard pile
    sz_disc = MAX_SIZE_DISC_PILE * dim_enc_card
    x_card_disc = x[:, idx : idx + sz_disc].view(batch_size, MAX_SIZE_DISC_PILE, dim_enc_card)
    idx += sz_disc

    # character
    x_char = x[:, idx : idx + dim_enc_char]
    idx += dim_enc_char

    # Monster
    x_monster = x[:, idx : idx + dim_enc_monster]
    idx += dim_enc_monster

    # Energy
    x_energy = x[:, idx : idx + dim_enc_energy]
    idx += dim_enc_energy

    return CombatViewEncoding(
        x_len_hand=x_len_hand,
        x_len_draw=x_len_draw,
        x_len_disc=x_len_disc,
        x_card_active_mask=x_card_active_mask,
        x_card_hand=x_card_hand,
        x_card_draw=x_card_draw,
        x_card_disc=x_card_disc,
        x_char=x_char,
        x_monster=x_monster,
        x_energy=x_energy,
    )
