# TODO: this is very tightly coupled with the models, improve it

from dataclasses import dataclass

import torch

from src.game.combat.effect import EffectType
from src.game.combat.view import CombatView
from src.game.combat.view.actor import ActorView
from src.game.combat.view.actor import ModifierViewType
from src.game.combat.view.card import CardView
from src.game.combat.view.character import CharacterView
from src.game.combat.view.energy import EnergyView
from src.game.combat.view.monster import IntentView
from src.game.combat.view.monster import MonsterView


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


# TODO: add batching?
@dataclass(frozen=True)
class Encoding:
    character: torch.Tensor
    monster: torch.Tensor
    energy: torch.Tensor
    hand: torch.Tensor

    idx_card_active: int | None
    effect: bool


# TODO: in the future I'll need to add max-energy, it's always 3 for now
def _encode_energy_view(energy_view: EnergyView, device: torch.device) -> torch.Tensor:
    return torch.tensor([energy_view.current], device=device, dtype=torch.float32)


def _encode_actor_modifiers(actor_view: ActorView) -> list[int]:
    modifier_encodings = [0] * len(MODIFIER_VIEW_TYPES)
    for i, modifier_view_type in enumerate(MODIFIER_VIEW_TYPES):
        if modifier_view_type in actor_view.modifiers:
            stacks_current = actor_view.modifiers[modifier_view_type]
            if stacks_current is None:
                modifier_encodings[i] = 1
            else:
                modifier_encodings[i] = stacks_current

    return modifier_encodings


def _encode_character_view(character_view: CharacterView, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            character_view.health_current,
            character_view.block_current,
            character_view.health_current + character_view.block_current,
            *_encode_actor_modifiers(character_view),
        ],
        dtype=torch.float32,
        device=device,
    )


# TODO: adapt to multiple monsters
def _encode_monster_view(monster_view: MonsterView, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            monster_view.health_current,
            monster_view.block_current,
            monster_view.health_current + monster_view.block_current,
            # Intent
            monster_view.intent.damage or 0,
            monster_view.intent.instances or 0,
            monster_view.intent.block,
            monster_view.intent.buff,
            # Modifiers
            *_encode_actor_modifiers(monster_view),
        ],
        dtype=torch.float32,
        device=device,
    )


# TODO: in the future, I should also encode the target and selection types
def _encode_card_view(card_view: CardView, energy_current: int) -> list[int]:
    card_encoded = [0] * len(EFFECT_TYPES)
    for effect in card_view.effects:
        card_encoded[EFFECT_TYPES.index(effect.type)] = effect.value

    return card_encoded + [card_view.cost, energy_current - card_view.cost]


def _encode_card_views(
    card_views: list[CardView], energy_current: int, device: torch.device
) -> torch.Tensor:
    return torch.tensor(
        [_encode_card_view(card_view, energy_current) for card_view in card_views],
        dtype=torch.float32,
        device=device,
    )


def _sort_card_views_and_return_mapping(
    card_views: list[CardView],
) -> tuple[list[CardView], dict[int, int]]:
    card_views_indexes = list(enumerate(card_views))
    card_views_indexes_sorted = sorted(card_views_indexes, key=lambda x: x[1].name)

    card_views_sorted = [x[1] for x in card_views_indexes_sorted]
    index_mapping = {
        idx_new: idx_old for idx_new, (idx_old, _) in enumerate(card_views_indexes_sorted)
    }

    return card_views_sorted, index_mapping


def get_character_encoding_dim() -> torch.Size:
    character_view_dummy = CharacterView("Dummy", 0, 0, 0, {})
    return _encode_character_view(character_view_dummy, torch.device("cpu")).shape


def get_monster_encoding_dim() -> torch.Size:
    monster_view_dummy = MonsterView("Dummy", 0, 0, 0, {}, 0, IntentView(None, None, False, False))
    return _encode_monster_view(monster_view_dummy, torch.device("cpu")).shape


def get_card_encoding_dim() -> int:
    card_view_dummy = CardView(0, "Dummy", [], 0, False)
    return len(_encode_card_view(card_view_dummy, 0))


def get_energy_encoding_dim() -> torch.Size:
    energy_view_dummy = EnergyView(0, 0)
    return _encode_energy_view(energy_view_dummy, torch.device("cpu")).shape


def encode_combat_view(
    combat_view: CombatView, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[int, int]]:
    hand_sorted, index_mapping = _sort_card_views_and_return_mapping(combat_view.hand)

    idx_card_active = None
    for i, card_view in enumerate(combat_view.hand):
        if card_view.is_active:
            idx_card_active = i
            break

    encoding = Encoding(
        _encode_character_view(combat_view.character, device),
        _encode_monster_view(combat_view.monsters[0], device),
        _encode_energy_view(combat_view.energy, device),
        _encode_card_views(hand_sorted, combat_view.energy.current, device),
        idx_card_active,
        True if combat_view.effect is not None else False,
    )

    return encoding, index_mapping
