import numpy as np
import torch
from slai import GameState
from slai import Potion
from slai import PotionName
from slai import PotionRarity
from slai import members

from src.rl.constants import MAX_POTION_REWARDS
from src.rl.constants import MAX_POTION_SLOTS
from src.rl.constants import MAX_SHOP_POTIONS
from src.rl.encoding.effect import ENCODING_DIM_EFFECTS
from src.rl.encoding.effect import encode_effects_into
from src.rl.types import Slice
from src.rl.types import SliceKind


# Order = fill order = Core's global-offset order
SLICE_POTIONS = [
    Slice(SliceKind.POTION_OWNED, MAX_POTION_SLOTS),
    Slice(SliceKind.POTION_REWARD, MAX_POTION_REWARDS),
    Slice(SliceKind.POTION_SHOP, MAX_SHOP_POTIONS),
]
NUM_POTION_TOKENS = sum(slice_.size for slice_ in SLICE_POTIONS)
SLICE_KIND_POTIONS = {slice_.kind for slice_ in SLICE_POTIONS}


_POTION_NAME_TO_IDX = {potion_name: i for i, potion_name in enumerate(members(PotionName))}
_POTION_RARITY_TO_IDX = {potion_rarity: i for i, potion_rarity in enumerate(members(PotionRarity))}

ENCODING_DIM_POTION = (
    len(_POTION_NAME_TO_IDX)  # Name OHE
    + len(_POTION_RARITY_TO_IDX)  # Rarity OHE
    + 1  # Requires target
    + 1  # Combat only
    + ENCODING_DIM_EFFECTS  # Per-EffectKind effect blocks
)

# Per-potion encoding cache keyed by name (rarity/target/effects are fixed by name)
_POTION_ROW_CACHE: dict[PotionName, np.ndarray] = {}
_POTION_ROW_CACHE_MAX = 100_000


def encode_potion_into(potion: Potion, pos: int, out: np.ndarray) -> int:
    # Name OHE
    out[pos + _POTION_NAME_TO_IDX[potion.name]] = 1.0
    pos += len(_POTION_NAME_TO_IDX)

    # Rarity OHE
    out[pos + _POTION_RARITY_TO_IDX[potion.rarity]] = 1.0
    pos += len(_POTION_RARITY_TO_IDX)

    # Scalars
    out[pos] = float(potion.requires_target)
    out[pos + 1] = float(potion.combat_only)
    pos += 2

    # Per-EffectKind effect blocks
    return encode_effects_into(potion.effects, pos, out)


def encode_potion_into_w_cache(potion: Potion, out: np.ndarray) -> None:
    """Write the full potion encoding into `out`, cached by name (its fields are fixed by name)."""
    row = _POTION_ROW_CACHE.get(potion.name)
    if row is None:
        row = np.zeros(ENCODING_DIM_POTION, dtype=np.float32)
        encode_potion_into(potion, 0, row)
        row.flags.writeable = False  # guard the cached master copy
        if len(_POTION_ROW_CACHE) >= _POTION_ROW_CACHE_MAX:
            _POTION_ROW_CACHE.clear()
        _POTION_ROW_CACHE[potion.name] = row
    out[:] = row


def _get_potion_entities(kind: SliceKind, game_state: GameState) -> list:
    match kind:
        case SliceKind.POTION_OWNED:
            return game_state.potions
        case SliceKind.POTION_REWARD:
            return (
                [game_state.reward.potion]
                if game_state.reward is not None and game_state.reward.potion is not None
                else []
            )
        case SliceKind.POTION_SHOP:
            return game_state.shop.potions if game_state.shop is not None else []
        case _:
            raise ValueError(f"not a potion slice: {kind}")


def encode_batch_potions(
    batch_game_state: list[GameState], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode all potion slices into one (B, NUM_POTION_TOKENS, dim) tensor + mask."""
    batch_size = len(batch_game_state)
    np_out = np.zeros((batch_size, NUM_POTION_TOKENS, ENCODING_DIM_POTION), dtype=np.float32)
    np_pad = np.zeros((batch_size, NUM_POTION_TOKENS), dtype=bool)

    for b, game_state in enumerate(batch_game_state):
        offset = 0
        for slice_ in SLICE_POTIONS:
            for i, potion in enumerate(_get_potion_entities(slice_.kind, game_state)):
                if potion is None:
                    continue

                encode_potion_into_w_cache(potion, np_out[b, offset + i])
                np_pad[b, offset + i] = True

            offset += slice_.size

    return (
        torch.from_numpy(np_out).to(device),
        torch.from_numpy(np_pad).to(device),
    )
