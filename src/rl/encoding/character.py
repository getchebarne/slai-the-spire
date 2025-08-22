import torch

from src.game.view.character import ViewCharacter
from src.rl.encoding.actor import encode_view_actor_modifiers
from src.rl.utils import encode_one_hot


_BLOCK_MAX = 20  # TODO: revisit
_BLOCK_MIN = 0
_HEALTH_MAX = 70  # TODO: will be dynamic in the future
_HEALTH_MIN = 1


def _get_view_character_dummy() -> ViewCharacter:
    return ViewCharacter("Dummy", 0, 0, 0, {}, 0)


def get_encoding_character_dim() -> int:
    view_character_dummy = _get_view_character_dummy()
    encoding_character_dummy = encode_view_character(view_character_dummy, torch.device("cpu"))
    return encoding_character_dummy.shape[0]


def encode_view_character(view_character: ViewCharacter, device: torch.device) -> torch.Tensor:
    return torch.cat(
        [
            encode_one_hot(view_character.health_current, _HEALTH_MIN, _HEALTH_MAX, device),
            encode_one_hot(view_character.block_current, _BLOCK_MIN, _BLOCK_MAX, device),
            encode_view_actor_modifiers(view_character.modifiers, device),
            torch.tensor(
                [
                    view_character.health_current / _HEALTH_MAX,
                    view_character.block_current / _BLOCK_MAX,
                ],
                dtype=torch.float32,
                device=device,
            ),
        ],
    )
