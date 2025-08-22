import torch

from src.game.const import CARD_REWARD_NUM
from src.game.const import MAP_HEIGHT
from src.game.const import MAP_WIDTH
from src.game.const import MAX_MONSTERS
from src.game.const import MAX_SIZE_DECK
from src.game.const import MAX_SIZE_DISC_PILE
from src.game.const import MAX_SIZE_DRAW_PILE
from src.game.const import MAX_SIZE_HAND
from src.game.view.state import ViewGameState
from src.rl.encoding.card import encode_view_cards
from src.rl.encoding.card import get_encoding_card_dim
from src.rl.encoding.character import encode_view_character
from src.rl.encoding.character import get_encoding_character_dim
from src.rl.encoding.energy import encode_view_energy
from src.rl.encoding.energy import get_encoding_energy_dim
from src.rl.encoding.map_ import encode_view_map
from src.rl.encoding.map_ import get_encoding_map_dim
from src.rl.encoding.monster import encode_view_monsters
from src.rl.encoding.monster import get_encoding_monster_dim


_ENCODING_DIM_CARD = get_encoding_card_dim()
_ENCODING_DIM_CHARACTER = get_encoding_character_dim()
_ENCODING_DIM_ENERGY = get_encoding_energy_dim()
_ENCODING_DIM_MAP = get_encoding_map_dim()
_ENCODING_DIM_MONSTER = get_encoding_monster_dim()

# TODO: revisit
_ENCODING_DIMENSIONS = {
    "hand": (MAX_SIZE_HAND, _ENCODING_DIM_CARD),
    "hand_pad": (MAX_SIZE_HAND,),
    "hand_active": (MAX_SIZE_HAND,),
    "draw": (MAX_SIZE_DRAW_PILE, _ENCODING_DIM_CARD),
    "draw_pad": (MAX_SIZE_DRAW_PILE,),
    "disc": (MAX_SIZE_DISC_PILE, _ENCODING_DIM_CARD),
    "disc_pad": (MAX_SIZE_DISC_PILE,),
    "reward": (CARD_REWARD_NUM, _ENCODING_DIM_CARD),
    "deck": (MAX_SIZE_DECK, _ENCODING_DIM_CARD),
    "char": (_ENCODING_DIM_CHARACTER,),
    "monster": (MAX_MONSTERS, _ENCODING_DIM_MONSTER),
    "energy": (_ENCODING_DIM_ENERGY,),
    "map": _ENCODING_DIM_MAP,
}
print(f"{_ENCODING_DIMENSIONS=}")


def encode_view_game_state(
    view_game_state: ViewGameState, device: torch.device
) -> tuple[torch.Tensor, ...]:
    # Encode cards in each collection
    x_hand, x_hand_pad, x_hand_active = encode_view_cards(
        view_game_state.hand, MAX_SIZE_HAND, device
    )
    x_draw, x_draw_pad, _ = encode_view_cards(
        view_game_state.pile_draw, MAX_SIZE_DRAW_PILE, device
    )
    x_disc, x_disc_pad, _ = encode_view_cards(
        view_game_state.pile_disc, MAX_SIZE_DISC_PILE, device
    )
    x_reward, _, _ = encode_view_cards(view_game_state.reward_combat, CARD_REWARD_NUM, device)
    x_deck, x_deck_pad, _ = encode_view_cards(view_game_state.deck, MAX_SIZE_DECK, device)

    # Encode character
    x_char = encode_view_character(view_game_state.character, device)

    # Encode monsters
    x_monster, x_monster_pad = encode_view_monsters(view_game_state.monsters, device=device)

    # Encode energy
    x_energy = encode_view_energy(view_game_state.energy, device)

    # Encode map
    x_map = encode_view_map(view_game_state.map, device)

    # Collect all tensors in a specific, consistent order
    return (
        x_hand.view(1, MAX_SIZE_HAND, -1),
        x_hand_pad.view(1, -1),
        x_hand_active.view(1, -1),
        x_draw.view(1, MAX_SIZE_DRAW_PILE, -1),
        x_draw_pad.view(1, -1),
        x_disc.view(1, MAX_SIZE_DISC_PILE, -1),
        x_disc_pad.view(1, -1),
        x_deck.view(1, MAX_SIZE_DECK, -1),
        x_deck_pad.view(1, -1),
        x_reward.view(1, CARD_REWARD_NUM, -1),
        x_char.view(1, -1),
        x_monster.view(1, MAX_MONSTERS, -1),
        x_monster_pad.view(1, -1),
        x_energy.view(1, -1),
        x_map.view(1, MAP_HEIGHT, MAP_WIDTH, -1),
    )


def unpack_encoded_state(x_state: torch.Tensor) -> dict[str, torch.Tensor]:
    x_all = {}

    # The order of keys must EXACTLY match the concatenation order in the encoder
    # This is why using a predefined structure like ENCODING_INFO is reliable.
    idx_start = 0
    for key, dim in _ENCODING_DIMENSIONS.items():
        # Calculate the number of elements for this component
        num_elements = torch.prod(torch.tensor(dim)).item()

        # Slice the flat tensor
        idx_end = idx_start + num_elements
        x_slice = x_state[idx_start:idx_end]

        # Reshape the slice back to its original shape and store it
        x_all[key] = x_slice.view(dim)

        # Move the index for the next slice
        idx_start = idx_end

    if idx_end != x_state.shape[0]:
        raise ValueError(
            f"{x_state.shape[0] - idx_end} elements were lost during state tensor unpacking"
        )

    return x_all
