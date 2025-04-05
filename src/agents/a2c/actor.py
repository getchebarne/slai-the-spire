import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.a2c.encode import Encoding
from src.agents.a2c.encode import _sort_card_views_and_return_mapping
from src.agents.a2c.encode import encode_combat_view
from src.agents.a2c.encode import get_card_encoding_dim
from src.agents.a2c.encode import get_character_encoding_dim
from src.agents.a2c.encode import get_energy_encoding_dim
from src.agents.a2c.encode import get_monster_encoding_dim
from src.agents.models.mlp import MLP
from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.constant import MAX_HAND_SIZE
from src.game.combat.view import CombatView


DIM_ENC_CARD = get_card_encoding_dim()
DIM_ENC_CHARACTER = get_character_encoding_dim()[0]
DIM_ENC_ENERGY = get_energy_encoding_dim()[0]
DIM_ENC_MONSTER = get_monster_encoding_dim()[0]
DIM_OTHER = DIM_ENC_CHARACTER + DIM_ENC_MONSTER + DIM_ENC_ENERGY


# TODO: parametrize dimensions better, so they are more readable
class Actor(nn.Module):
    def __init__(self, dim_card: int):
        super().__init__()

        self._dim_card = dim_card

        # Card embedding
        self._embedding_card_pad = nn.Parameter(torch.randn(DIM_ENC_CARD))
        self._embedding_card = nn.Linear(DIM_ENC_CARD, dim_card)

        # Other state variables (i.e., character, monsters, energy)
        self._embedding_other = MLP([DIM_OTHER, 3 * dim_card, 3 * dim_card])

        # Two actions for every card (select/play card, discard card) for now
        self._mlp_card = nn.Sequential(
            MLP([11 * dim_card, 11 * dim_card, 11 * dim_card]),
            nn.Linear(11 * dim_card, 2),
        )

        # One action for every monster (only one monster for now) + 1 for end-of-turn
        self._mlp_monster_end_turn = nn.Sequential(
            MLP([10 * dim_card, 10 * dim_card, 10 * dim_card]),
            nn.Linear(10 * dim_card, 2),
        )

    def forward(self, encoding: Encoding, x_valid_action_mask: torch.Tensor) -> torch.Tensor:
        # Hand
        x_card_hand = torch.cat(
            [
                encoding.hand,
                self._embedding_card_pad.unsqueeze(0).expand(
                    MAX_HAND_SIZE - len(encoding.hand), DIM_ENC_CARD
                ),
            ]
        )
        x_card_hand = self._embedding_card(x_card_hand)

        # Other
        x_other = self._embedding_other(
            torch.cat(
                [
                    encoding.character,
                    encoding.monster,
                    encoding.energy,
                ]
            )
        )

        # Concatenate
        x_global = torch.cat(
            [
                torch.flatten(x_card_hand),
                x_other,
            ]
        )

        # Calculate logits
        x_logit_cards = self._mlp_card(
            torch.cat([x_global.expand((MAX_HAND_SIZE, 10 * self._dim_card)), x_card_hand], dim=1)
        )
        x_logit_monster_end_turn = self._mlp_monster_end_turn(x_global)
        x_logit_all = torch.cat(
            [
                torch.flatten(x_logit_cards.permute(1, 0)),
                x_logit_monster_end_turn,
            ]
        )

        # Apply valid action mask and get action probabilities w/ softmax
        x_logit_all_mask = x_logit_all.masked_fill(x_valid_action_mask == 0, float("-inf"))
        x_prob = F.softmax(x_logit_all_mask, dim=-1)

        return x_prob


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
    hand_sorted, _ = _sort_card_views_and_return_mapping(combat_view.hand)
    valid_action_mask = [card.cost <= combat_view.energy.current for card in hand_sorted]
    valid_action_mask.extend([False] * (MAX_HAND_SIZE - len(combat_view.hand)))
    valid_action_mask.extend([False] * MAX_HAND_SIZE)
    valid_action_mask.extend([False, True])

    return valid_action_mask


# TODO: adapt for multiple monsters
# TODO: adapt for discard
def action_idx_to_action(
    action_idx: int, combat_view: CombatView, index_mapping: dict[int, int]
) -> Action:
    if action_idx < 2 * MAX_HAND_SIZE:
        # Undo hand sorting TODO: improve
        action_idx = index_mapping[action_idx % MAX_HAND_SIZE]
        return Action(ActionType.SELECT_ENTITY, combat_view.hand[action_idx].entity_id)

    if action_idx == 2 * MAX_HAND_SIZE:
        return Action(ActionType.SELECT_ENTITY, combat_view.monsters[0].entity_id)

    if action_idx == 2 * MAX_HAND_SIZE + 1:
        return Action(ActionType.END_TURN)

    raise ValueError(f"Unsupported action index: {action_idx}")


# TODO: return metadata
def select_action(
    model: Actor,
    combat_view: CombatView,
    greedy: bool = True,
    device: torch.device = torch.device("cpu"),
) -> tuple[Action, torch.Tensor, torch.Tensor]:
    encoding, index_mapping = encode_combat_view(combat_view, device)
    valid_action_mask = get_valid_action_mask(combat_view)

    probs = model(encoding, torch.tensor(valid_action_mask, dtype=torch.bool, device=device))
    if greedy:
        action_idx = torch.argmax(probs)

    else:
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()

    action = action_idx_to_action(action_idx.item(), combat_view, index_mapping)

    return action
