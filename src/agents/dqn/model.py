import torch
import torch.nn as nn

from src.agents.dqn.encode import encode_combat_view
from src.agents.dqn.encode import get_card_encoding_dim
from src.agents.dqn.encode import get_character_encoding_dim
from src.agents.dqn.encode import get_energy_encoding_dim
from src.agents.dqn.encode import get_monster_encoding_dim
from src.agents.models.mlp import MLP
from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.constant import MAX_HAND_SIZE
from src.game.combat.view import CombatView


DIM_ENC_CARD = get_card_encoding_dim()
DIM_ENC_CHARACTER = get_character_encoding_dim()[0]
DIM_ENC_ENERGY = get_energy_encoding_dim()[0]
DIM_ENC_MONSTER = get_monster_encoding_dim()[0]
DIM_ENC_OTHER = DIM_ENC_CHARACTER + DIM_ENC_MONSTER + DIM_ENC_ENERGY


# TODO: parametrize dimensions better, so they are more readable
class DeepQNetwork(nn.Module):
    def __init__(self, dim_card: int):
        super().__init__()

        self._dim_card = dim_card

        # Card embedding
        self._embedding_card_pad = nn.Parameter(torch.randn(DIM_ENC_CARD))
        self._embedding_card = nn.Linear(DIM_ENC_CARD, dim_card)

        # Character embedding
        self._embedding_char = MLP([DIM_ENC_CHARACTER, DIM_ENC_CHARACTER, DIM_ENC_CHARACTER])

        # Monster embedding
        self._embedding_monster = MLP([DIM_ENC_MONSTER, DIM_ENC_MONSTER, DIM_ENC_MONSTER])

        # Energy embedding
        self._embedding_energy = MLP([DIM_ENC_ENERGY, DIM_ENC_ENERGY, DIM_ENC_ENERGY])

        # Layer normalization after concatenating other and cards
        self._ln_global = nn.LayerNorm(DIM_ENC_OTHER + 3 * dim_card)

        # Two actions for every card (select/play card, discard card) for now
        aux = DIM_ENC_OTHER + 4 * dim_card
        self._mlp_card = nn.Sequential(
            MLP([aux, aux, aux]),
            nn.Linear(aux, 2),
        )

        # One action for every monster (only one monster for now) + 1 for end-of-turn
        self._mlp_monster = nn.Sequential(
            MLP([aux, aux, aux]),
            nn.Linear(aux, 1),
        )
        self._mlp_end_turn = nn.Sequential(
            MLP([aux - dim_card, aux - dim_card, aux - dim_card]),
            nn.Linear(aux - dim_card, 1),
        )

    def forward(self, x_state: torch.Tensor) -> torch.Tensor:
        batch_size = x_state.shape[0]

        # Hand
        x_hand_size = x_state[:, 0:1]
        x_card_is_active = x_state[:, 1 : 1 + MAX_HAND_SIZE]
        x_hand = x_state[
            :, 1 + MAX_HAND_SIZE : 1 + MAX_HAND_SIZE + DIM_ENC_CARD * MAX_HAND_SIZE
        ].view(batch_size, MAX_HAND_SIZE, DIM_ENC_CARD)
        high = 1 + MAX_HAND_SIZE + DIM_ENC_CARD * MAX_HAND_SIZE

        # Other
        x_char = x_state[:, high : high + DIM_ENC_CHARACTER]
        high += DIM_ENC_CHARACTER
        x_monster = x_state[:, high : high + DIM_ENC_MONSTER]
        high += DIM_ENC_MONSTER
        x_energy = x_state[:, high:]

        # Pad hand
        mask_pad = torch.arange(MAX_HAND_SIZE).view(1, MAX_HAND_SIZE).expand(
            batch_size, MAX_HAND_SIZE
        ) >= x_hand_size.expand(batch_size, MAX_HAND_SIZE)
        mask_pad = mask_pad.view(batch_size, MAX_HAND_SIZE, 1).expand(
            batch_size, MAX_HAND_SIZE, DIM_ENC_CARD
        )
        x_hand[mask_pad] = self._embedding_card_pad.view(1, 1, DIM_ENC_CARD).expand(
            batch_size, MAX_HAND_SIZE, DIM_ENC_CARD
        )[mask_pad]

        # Encode hand
        x_card_hand = self._embedding_card(x_hand)

        # Other (monster, character, energy)
        x_monster = self._embedding_monster(x_monster)
        x_char = self._embedding_char(x_char)
        x_energy = self._embedding_energy(x_energy)

        # Concatenate
        x_global = self._ln_global(
            torch.cat(
                [
                    torch.sum(x_card_hand, dim=1),
                    torch.mean(x_card_hand, dim=1),
                    torch.max(x_card_hand, dim=1)[0],
                    x_monster,
                    x_char,
                    x_energy,
                ],
                dim=1,
            )
        )

        # Calculate actions
        x_card = self._mlp_card(
            torch.cat(
                [
                    x_global.view(batch_size, 1, -1).expand(batch_size, MAX_HAND_SIZE, -1),
                    x_card_hand,
                ],
                dim=2,
            )
        )
        x_card_active = torch.sum(
            (
                x_card_is_active.view(batch_size, MAX_HAND_SIZE, 1).expand(
                    batch_size, MAX_HAND_SIZE, self._dim_card
                )
                * x_card_hand
            ),
            dim=1,
        )
        x_monster = self._mlp_monster(
            torch.cat(
                [
                    x_global,
                    x_card_active,
                ],
                dim=1,
            )
        )
        x_end_turn = self._mlp_end_turn(x_global)

        return torch.cat(
            [
                torch.flatten(x_card.permute(0, 2, 1), start_dim=1),
                x_monster,
                x_end_turn,
            ],
            dim=1,
        )


# TODO: must change when more monsters are added
# TODO: make more readable
def get_valid_action_mask(combat_view: CombatView) -> list[bool]:
    if any(card.is_active for card in combat_view.hand):
        # Only valid action is to select the monster
        return [False] * (2 * MAX_HAND_SIZE) + [True, False]

    if combat_view.effect is not None:
        # TODO: only contemplating EffectType.DISCARD for now
        valid_action_mask = [False] * MAX_HAND_SIZE
        valid_action_mask.extend([True] * len(combat_view.hand))
        valid_action_mask.extend([False] * (MAX_HAND_SIZE - len(combat_view.hand)))
        valid_action_mask.extend([False, False])

        return valid_action_mask

    # Undo hand sorting TODO: improve
    valid_action_mask = [card.cost <= combat_view.energy.current for card in combat_view.hand]
    valid_action_mask.extend([False] * (MAX_HAND_SIZE - len(combat_view.hand)))
    valid_action_mask.extend([False] * MAX_HAND_SIZE)
    valid_action_mask.extend([False, True])

    return valid_action_mask


# TODO: adapt for multiple monsters
# TODO: adapt for discard
def action_idx_to_action(action_idx: int, combat_view: CombatView) -> Action:
    if action_idx < 2 * MAX_HAND_SIZE:
        # Undo hand sorting TODO: improve
        return Action(
            ActionType.SELECT_ENTITY, combat_view.hand[action_idx % MAX_HAND_SIZE].entity_id
        )

    if action_idx == 2 * MAX_HAND_SIZE:
        return Action(ActionType.SELECT_ENTITY, combat_view.monsters[0].entity_id)

    if action_idx == 2 * MAX_HAND_SIZE + 1:
        return Action(ActionType.END_TURN)

    raise ValueError(f"Unsupported action index: {action_idx}")


def select_action(
    model: DeepQNetwork,
    combat_view: CombatView,
    device: torch.device = torch.device("cpu"),
) -> tuple[Action, torch.Tensor]:
    x_state = encode_combat_view(combat_view, device)
    valid_action_mask_tensor = torch.tensor(
        [get_valid_action_mask(combat_view)], dtype=torch.bool, device=device
    )

    with torch.no_grad():
        q_values = model(x_state.unsqueeze(0))

    q_values[~valid_action_mask_tensor] = float("-inf")
    action_idx = torch.argmax(q_values).item()
    action = action_idx_to_action(action_idx, combat_view)

    return action, q_values
