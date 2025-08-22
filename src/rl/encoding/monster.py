import torch

from src.game.const import MAX_MONSTERS
from src.game.factory.lib import FACTORY_LIB_MONSTER
from src.game.factory.monster.the_guardian import _FIERCE_BASH_DAMAGE_ASC_4  # TODO: queries
from src.game.factory.monster.the_guardian import _HEALTH_MAX_ASC_9  # TODO: queries
from src.game.factory.monster.the_guardian import _WHIRLWIND_INSTANCES  # TODO: queries
from src.game.view.monster import ViewIntent
from src.game.view.monster import ViewMonster
from src.rl.encoding.actor import encode_view_actor_modifiers
from src.rl.utils import encode_one_hot


_BLOCK_MAX = 20  # TODO: revisit
_BLOCK_MIN = 0
_DAMAGE_MAX = _FIERCE_BASH_DAMAGE_ASC_4
_HEALTH_MAX = _HEALTH_MAX_ASC_9
_HEALTH_MIN = 1
_INSTANCES_MAX = _WHIRLWIND_INSTANCES


def _get_monster_names() -> list[str]:
    return list(FACTORY_LIB_MONSTER.keys())


_MONSTER_NAMES = _get_monster_names()


def _encode_view_monster(view_monster: ViewMonster, device: torch.device) -> torch.Tensor:
    idx_name = _MONSTER_NAMES.index(view_monster.name)
    return torch.cat(
        [
            encode_one_hot(idx_name, 0, len(_MONSTER_NAMES) - 1, device),
            encode_one_hot(view_monster.health_current, _HEALTH_MIN, _HEALTH_MAX, device),
            encode_one_hot(view_monster.block_current, _BLOCK_MIN, _BLOCK_MAX, device),
            encode_view_actor_modifiers(view_monster.modifiers, device),
            _encode_view_intent(view_monster.intent, device),
            torch.tensor(
                [
                    view_monster.health_current / _HEALTH_MAX,
                    view_monster.block_current / _BLOCK_MAX,
                ],
                dtype=torch.float32,
                device=device,
            ),
        ],
    )


def _encode_view_intent(view_intent: ViewIntent, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            (view_intent.damage or 0) / _DAMAGE_MAX,
            (view_intent.instances or 0) / _INSTANCES_MAX,
            view_intent.block,
            view_intent.buff,
            view_intent.debuff_powerful,
        ],
        dtype=torch.float32,
        device=device,
    )


def _get_view_monster_dummy() -> ViewMonster:
    return ViewMonster("Dummy", 0, 0, 0, {}, ViewIntent())


def get_encoding_monster_dim() -> int:
    view_monster_dummy = _get_view_monster_dummy()
    encoding_monster_dummy = _encode_view_monster(view_monster_dummy, torch.device("cpu"))
    return encoding_monster_dummy.shape[0]


def encode_view_monsters(
    view_monsters: list[ViewMonster], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    mask_pad = torch.arange(MAX_MONSTERS, dtype=torch.float32, device=device) < len(view_monsters)
    if not view_monsters:
        # Get monster encoding dimension
        encoding_monster_dim = get_encoding_monster_dim()

        # Return all-zeros tensor of shape (`MAX_MONSTERS`, `monster_encoding_dim`)
        return (
            torch.zeros(MAX_MONSTERS, encoding_monster_dim, dtype=torch.float32, device=device),
            mask_pad,
        )

    encoding_monsters = None
    for idx, view_monster in enumerate(view_monsters):
        # Get encoding
        encoding_monster = _encode_view_monster(view_monster, device)

        if encoding_monsters is None:
            # Intialize all-zeros tensor to hold all encodings, now that we now the encoding dimension
            encoding_monsters = torch.zeros(
                MAX_MONSTERS, encoding_monster.shape[0], dtype=torch.float32, device=device
            )

        # Assign encoding
        encoding_monsters[idx] = encoding_monster

    return encoding_monsters, mask_pad
