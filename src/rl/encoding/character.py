import torch

from src.game.factory.monster.the_guardian import _FIERCE_BASH_DAMAGE_ASC_4
from src.game.view.character import ViewCharacter
from src.rl.encoding.actor import encode_view_actor_modifiers
from src.rl.utils import encode_one_hot_list


_BLOCK_MAX = 20  # TODO: revisit
_BLOCK_MIN = 0
_HEALTH_MAX = 70  # TODO: will be dynamic in the future
_HEALTH_MIN = 1


def get_encoding_dim_character() -> int:
    view_character_dummy = ViewCharacter("Dummy", 0, 0, 0, {}, 0)
    encoding_character_dummy = _encode_view_character(view_character_dummy, 0)
    return len(encoding_character_dummy)


def _encode_view_character(view_character: ViewCharacter, incoming_damage: int) -> list[float]:
    return (
        # Health / one-hot
        encode_one_hot_list(view_character.health_current, _HEALTH_MIN, _HEALTH_MAX)
        # Block / one-hot
        + encode_one_hot_list(view_character.block_current, _BLOCK_MIN, _BLOCK_MAX)
        # Health + Block / one-hot
        + encode_one_hot_list(
            view_character.health_current + view_character.block_current,
            _HEALTH_MIN + _BLOCK_MIN,
            _HEALTH_MAX + _BLOCK_MAX,
        )
        # Modifiers
        + encode_view_actor_modifiers(view_character.modifiers)
        # Scalars
        + [
            # Health / scalar
            view_character.health_current / _HEALTH_MAX,
            # Block / scalar
            view_character.block_current / _BLOCK_MAX,
            # Health + Block / scalar
            (view_character.health_current + view_character.block_current)
            / (_HEALTH_MAX + _BLOCK_MAX),
            # Incoming damage
            incoming_damage / _FIERCE_BASH_DAMAGE_ASC_4,
            # Wether incoming damage is blocked or not
            float(view_character.block_current >= incoming_damage),
        ]
    )


def encode_batch_view_character(
    batch_view_character: list[ViewCharacter],
    batch_incoming_damage: list[int],
    device: torch.device,
) -> torch.Tensor:
    encodings_list = []

    # Iterate through each character and its corresponding damage in the batch
    for view_character, incoming_damage in zip(batch_view_character, batch_incoming_damage):
        # Call the single-item encoder for each item
        encoding = _encode_view_character(view_character, incoming_damage)
        encodings_list.append(encoding)

    return torch.tensor(encodings_list, dtype=torch.float32, device=device)
