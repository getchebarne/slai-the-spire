from enum import StrEnum
from enum import auto

import numpy as np
import torch
from slai import Card
from slai import Character

from src.rl.constants import BLOCK_MAX_ENTITY
from src.rl.constants import HEALTH_MAX_ENTITY
from src.rl.constants import MAX_SIZE_DECK
from src.rl.encoding.health_block import encode_health_block_into
from src.rl.encoding.health_block import get_encoding_dim_health_block
from src.rl.types import Slice
from src.rl.types import SliceKind
from src.rl.types import make_channel_allocator
from src.rl.utils import get_piecewise_bucket
from src.rl.utils import get_piecewise_dim
from src.rl.utils import get_sqrt_norm


# Order = fill order = Core's global-offset order
SLICE_CHARACTER = [Slice(SliceKind.CHARACTER, 1)]


_INCOMING_DAMAGE_MAX = 150
_HEALTH_MAX = HEALTH_MAX_ENTITY
_HEALTH_MAX_CAP = 200
_BLOCK_MAX = BLOCK_MAX_ENTITY
_GOLD_MIN = 0
_GOLD_MAX = 499
_GOLD_LINEAR_SQRT_THRESHOLD = _GOLD_MIN  # Pure sqrt
_GOLD_DIM = get_piecewise_dim(_GOLD_MIN, _GOLD_MAX, _GOLD_LINEAR_SQRT_THRESHOLD)
_DIM_HEALTH_BLOCK = get_encoding_dim_health_block(_HEALTH_MAX, _BLOCK_MAX)


class CharacterChannel(StrEnum):
    HEALTH_BLOCK = auto()
    GOLD = auto()
    GOLD_SCALAR = auto()
    INCOMING_DAMAGE = auto()
    BLOCK_COVERS_INCOMING = auto()
    INCOMING_IS_LETHAL = auto()
    HEALTH_FRACTION = auto()
    MAX_HEALTH_MAGNITUDE = auto()
    NET_UNBLOCKED_DAMAGE = auto()
    DECK_SIZE = auto()
    DECK_UPGRADED_FRACTION = auto()


CHANNEL_LAYOUT_CHARACTER, _alloc = make_channel_allocator()
OFF_CHARACTER_HEALTH_BLOCK = _alloc(CharacterChannel.HEALTH_BLOCK, _DIM_HEALTH_BLOCK)
OFF_CHARACTER_GOLD = _alloc(CharacterChannel.GOLD, _GOLD_DIM)
OFF_CHARACTER_GOLD_SCALAR = _alloc(CharacterChannel.GOLD_SCALAR)
OFF_CHARACTER_INCOMING_DAMAGE = _alloc(CharacterChannel.INCOMING_DAMAGE)
OFF_CHARACTER_BLOCK_COVERS_INCOMING = _alloc(CharacterChannel.BLOCK_COVERS_INCOMING)
OFF_CHARACTER_INCOMING_IS_LETHAL = _alloc(CharacterChannel.INCOMING_IS_LETHAL)
OFF_CHARACTER_HEALTH_FRACTION = _alloc(CharacterChannel.HEALTH_FRACTION)
OFF_CHARACTER_MAX_HEALTH_MAGNITUDE = _alloc(CharacterChannel.MAX_HEALTH_MAGNITUDE)
OFF_CHARACTER_NET_UNBLOCKED_DAMAGE = _alloc(CharacterChannel.NET_UNBLOCKED_DAMAGE)
OFF_CHARACTER_DECK_SIZE = _alloc(CharacterChannel.DECK_SIZE)
OFF_CHARACTER_DECK_UPGRADED_FRACTION = _alloc(CharacterChannel.DECK_UPGRADED_FRACTION)

ENCODING_DIM_CHARACTER = sum(w for _, w in CHANNEL_LAYOUT_CHARACTER)
SLICE_HEALTH_BLOCK_CHARACTER = slice(
    OFF_CHARACTER_HEALTH_BLOCK, OFF_CHARACTER_HEALTH_BLOCK + _DIM_HEALTH_BLOCK
)


def _encode_character_into(
    character: Character, incoming_damage: int, deck: list[Card], out: np.ndarray
) -> None:
    encode_health_block_into(
        character.health, character.block, _HEALTH_MAX, _BLOCK_MAX, OFF_CHARACTER_HEALTH_BLOCK, out
    )

    gold_bucket = get_piecewise_bucket(
        character.gold, _GOLD_MIN, _GOLD_MAX, _GOLD_LINEAR_SQRT_THRESHOLD
    )
    out[OFF_CHARACTER_GOLD + gold_bucket] = 1.0

    out[OFF_CHARACTER_GOLD_SCALAR] = min(character.gold, _GOLD_MAX) / _GOLD_MAX
    out[OFF_CHARACTER_INCOMING_DAMAGE] = get_sqrt_norm(incoming_damage, _INCOMING_DAMAGE_MAX)
    out[OFF_CHARACTER_BLOCK_COVERS_INCOMING] = float(character.block >= incoming_damage)
    out[OFF_CHARACTER_INCOMING_IS_LETHAL] = float(
        incoming_damage >= character.health + character.block
    )
    out[OFF_CHARACTER_HEALTH_FRACTION] = character.health / max(character.health_max, 1)
    out[OFF_CHARACTER_MAX_HEALTH_MAGNITUDE] = get_sqrt_norm(character.health_max, _HEALTH_MAX_CAP)
    out[OFF_CHARACTER_NET_UNBLOCKED_DAMAGE] = get_sqrt_norm(
        max(incoming_damage - character.block, 0), _INCOMING_DAMAGE_MAX
    )
    out[OFF_CHARACTER_DECK_SIZE] = get_sqrt_norm(len(deck), MAX_SIZE_DECK)
    out[OFF_CHARACTER_DECK_UPGRADED_FRACTION] = sum(card.upgraded for card in deck) / max(
        len(deck), 1
    )


def encode_batch_character(
    batch_character: list[Character],
    batch_incoming_damage: list[int],
    batch_deck: list[list[Card]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(batch_character)

    # Pre-allocate NumPy array
    np_out = np.zeros((batch_size, ENCODING_DIM_CHARACTER), dtype=np.float32)

    for b, (character, incoming_damage, deck) in enumerate(
        zip(batch_character, batch_incoming_damage, batch_deck)
    ):
        _encode_character_into(character, incoming_damage, deck, np_out[b])

    # Singleton entity (always present): a seq dim of 1 + an all-valid mask, for a TPadded slice
    return torch.from_numpy(np_out).to(device).unsqueeze(1), torch.ones(
        batch_size, 1, dtype=torch.bool, device=device
    )
