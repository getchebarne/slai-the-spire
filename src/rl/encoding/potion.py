import numpy as np
import torch
from slai import GameState
from slai import Potion
from slai import PotionName
from slai import PotionRarity
from slai import members

from src.rl.encoding.effect import ENCODING_DIM_EFFECTS
from src.rl.encoding.effect import encode_effects_into
from src.rl.index import KIND_TOKENS
from src.rl.index import LOCAL_SLICE
from src.rl.index import NUM_KIND_TOKENS
from src.rl.index import TokenKind
from src.rl.index import token_entities


_POTION_NAME_TO_IDX = {potion_name: i for i, potion_name in enumerate(members(PotionName))}
_POTION_RARITY_TO_IDX = {potion_rarity: i for i, potion_rarity in enumerate(members(PotionRarity))}

ENCODING_DIM_POTION = (
    len(_POTION_NAME_TO_IDX)  # Name OHE
    + len(_POTION_RARITY_TO_IDX)  # Rarity OHE
    + 1  # Requires target
    + 1  # Combat only
    + ENCODING_DIM_EFFECTS  # Per-EffectKind effect blocks
)


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


def encode_batch_potions(
    batch_game_state: list[GameState], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode every potion token (registry POTION kind) into one concatenated
    (B, N_POTIONS, ENCODING_DIM_POTION) tensor + mask; tokens live at their
    index.LOCAL_SLICE positions. Belt slots may hold None mid-list (drunk potion);
    the slot keeps its index and stays masked. Targeting is no longer derived
    here — the L3 target mask comes from the engine's legal actions (masks.py)."""
    batch_size = len(batch_game_state)
    num_tokens = NUM_KIND_TOKENS[TokenKind.POTION]

    # Pre-allocate NumPy arrays
    x_out = np.zeros((batch_size, num_tokens, ENCODING_DIM_POTION), dtype=np.float32)
    x_pad = np.zeros((batch_size, num_tokens), dtype=bool)

    for b, game_state in enumerate(batch_game_state):
        for token in KIND_TOKENS[TokenKind.POTION]:
            offset = LOCAL_SLICE[token].start
            for i, potion in enumerate(token_entities(token, game_state)):
                if potion is None:
                    continue

                encode_potion_into(potion, 0, x_out[b, offset + i])
                x_pad[b, offset + i] = True

    return (
        torch.from_numpy(x_out).to(device),
        torch.from_numpy(x_pad).to(device),
    )
