import torch
import torch.nn as nn
import torch.nn.functional as F

from src.game.combat.constant import MAX_HAND_SIZE
from src.rl.encoding import get_card_encoding_dim
from src.rl.encoding import get_character_encoding_dim
from src.rl.encoding import get_energy_encoding_dim
from src.rl.encoding import get_monster_encoding_dim


DIM_ENC_CARD = get_card_encoding_dim()
DIM_ENC_CHARACTER = get_character_encoding_dim()[0]
DIM_ENC_ENERGY = get_energy_encoding_dim()[0]
DIM_ENC_MONSTER = get_monster_encoding_dim()[0]
DIM_OTHER = DIM_ENC_CHARACTER + DIM_ENC_MONSTER + DIM_ENC_ENERGY


# TODO: implement this shi
class ActorCritic(nn.Module):
    def __init__(self, dim_card: int):
        super().__init__()

        self._dim_card = dim_card

        self._dummy_param = nn.Parameter(torch.randn(2 * MAX_HAND_SIZE + 2))

    def forward(
        self, x_state: torch.Tensor, x_valid_action_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.randn((1, 2 * MAX_HAND_SIZE + 2), dtype=torch.float32) * self._dummy_param
        value = torch.randn((1, 1), dtype=torch.float32)

        probs[~x_valid_action_mask.view(1, -1)] = float("-inf")

        return F.softmax(probs), value
