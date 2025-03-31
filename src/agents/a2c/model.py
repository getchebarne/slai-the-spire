import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.a2c.encode import Encoding
from src.agents.a2c.encode import encode_combat_view
from src.agents.a2c.encode import get_card_encoding_dim
from src.agents.a2c.encode import get_character_encoding_dim
from src.agents.a2c.encode import get_energy_encoding_dim
from src.agents.a2c.encode import get_monster_encoding_dim
from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.constant import MAX_HAND_SIZE
from src.game.combat.view import CombatView


DIM_ENC_CARD = get_card_encoding_dim()
DIM_ENC_CHARACTER = get_character_encoding_dim()[0]
DIM_ENC_ENERGY = get_energy_encoding_dim()[0]
DIM_ENC_MONSTER = get_monster_encoding_dim()[0]
DIM_OTHER = DIM_ENC_CHARACTER + DIM_ENC_MONSTER + DIM_ENC_ENERGY


# TODO: improve this, layer_sizes is not actually layer sizes
class MLP(nn.Module):
    def __init__(self, layer_sizes: list[int], bias: bool = True, activation_name: str = "ReLU"):
        super().__init__()

        self._layer_sizes = layer_sizes
        self._bias = bias
        self._activation_name = activation_name

        self._mlp = self._create()

    def _create(self) -> nn.Module:
        layers = []
        for i in range(len(self._layer_sizes) - 1):
            layers.append(nn.Linear(self._layer_sizes[i], self._layer_sizes[i + 1], self._bias))
            layers.append(getattr(nn, self._activation_name)())

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._mlp(x)


class ActorCritic(nn.Module):
    def __init__(self, dim_card: int):
        super().__init__()

        self._dim_card = dim_card

        # Card embedding and transformer
        self._embedding_card = nn.Linear(DIM_ENC_CARD, dim_card)

        # Other state variables (i.e., character, monsters, energy)
        self._embedding_other = MLP([DIM_OTHER, 3 * dim_card, 3 * dim_card])

        # Global layer normalization
        self._ln_global = nn.LayerNorm(12 * dim_card)

        # Actor
        self._ln_mlp_actor_card = nn.LayerNorm(13 * dim_card)
        self._mlp_actor_card = nn.Sequential(
            MLP([13 * dim_card, 13 * dim_card, 13 * dim_card]),
            nn.Linear(13 * dim_card, 2),
        )
        self._mlp_actor_monster = nn.Sequential(
            MLP([12 * dim_card, 12 * dim_card, 12 * dim_card]),
            nn.Linear(12 * dim_card, 1),
        )
        self._mlp_actor_end_turn = nn.Sequential(
            MLP([12 * dim_card, 12 * dim_card, 12 * dim_card]),
            nn.Linear(12 * dim_card, 1),
        )

        # Critic
        self._mlp_critic = nn.Sequential(
            MLP([12 * dim_card, 12 * dim_card, 12 * dim_card]),
            nn.Linear(12 * dim_card, 1),
        )

    def forward(
        self, x_state: Encoding, x_valid_action_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Hand
        x_card_hand_mask = torch.arange(MAX_HAND_SIZE) < x_state.hand_size
        x_card_hand_mask = x_card_hand_mask.view((MAX_HAND_SIZE, 1)).expand(
            (MAX_HAND_SIZE, self._dim_card)
        )
        x_card_hand = self._embedding_card(x_state.hand) * x_card_hand_mask

        # Draw pile
        draw_pile_size = x_state.draw_pile.shape[0]
        if draw_pile_size > 0:
            x_card_draw_pile = self._embedding_card(x_state.draw_pile)
            x_card_draw_pile = torch.cat(
                [
                    torch.mean(x_card_draw_pile, dim=0),
                    torch.sum(x_card_draw_pile, dim=0),
                    torch.max(x_card_draw_pile, dim=0)[0],
                ]
            )
        else:
            x_card_draw_pile = torch.zeros(3 * self._dim_card)

        # Discard pile
        disc_pile_size = x_state.disc_pile.shape[0]
        if disc_pile_size > 0:
            x_card_disc_pile = self._embedding_card(x_state.disc_pile)
            x_card_disc_pile = torch.cat(
                [
                    torch.mean(x_card_disc_pile, dim=0),
                    torch.sum(x_card_disc_pile, dim=0),
                    torch.max(x_card_disc_pile, dim=0)[0],
                ]
            )
        else:
            x_card_disc_pile = torch.zeros(3 * self._dim_card)

        # Other
        x_other = self._embedding_other(
            torch.cat(
                [
                    x_state.character,
                    x_state.monster,
                    x_state.energy,
                ]
            )
        )

        # Concatenate
        x_global = torch.cat(
            [
                torch.mean(x_card_hand, dim=0),
                torch.sum(x_card_hand, dim=0),
                torch.max(x_card_hand, dim=0)[0],
                x_card_draw_pile,
                x_card_disc_pile,
                x_other,
            ]
        )
        x_global = self._ln_global(x_global)

        # Actor
        log_probs_card = self._mlp_actor_card(
            self._ln_mlp_actor_card(
                torch.cat(
                    [x_global.expand((MAX_HAND_SIZE, 12 * self._dim_card)), x_card_hand], dim=1
                )
            )
        )
        log_probs_monster = self._mlp_actor_monster(x_global)
        log_probs_end_turn = self._mlp_actor_end_turn(x_global)
        log_probs = torch.cat(
            [
                torch.flatten(log_probs_card.permute(1, 0)),
                log_probs_monster,
                log_probs_end_turn,
            ]
        )
        log_probs_mask = log_probs.masked_fill(x_valid_action_mask == 0, float("-inf"))
        probs = F.softmax(log_probs_mask, dim=-1)

        # Critic
        value = self._mlp_critic(x_global)

        return probs, value


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

    valid_action_mask = [card.cost <= combat_view.energy.current for card in combat_view.hand]
    valid_action_mask.extend([False] * (MAX_HAND_SIZE - len(combat_view.hand)))
    valid_action_mask.extend([False] * MAX_HAND_SIZE)
    valid_action_mask.extend([False, True])

    return valid_action_mask


# TODO: adapt for multiple monsters
# TODO: adapt for discard
def action_idx_to_action(action_idx: int, combat_view: CombatView) -> Action:
    if action_idx < 2 * MAX_HAND_SIZE:
        return Action(
            ActionType.SELECT_ENTITY, combat_view.hand[action_idx % MAX_HAND_SIZE].entity_id
        )

    if action_idx == 2 * MAX_HAND_SIZE:
        return Action(ActionType.SELECT_ENTITY, combat_view.monsters[0].entity_id)

    if action_idx == 2 * MAX_HAND_SIZE + 1:
        return Action(ActionType.END_TURN)

    raise ValueError(f"Unsupported action index: {action_idx}")


def select_action(
    model: ActorCritic,
    combat_view: CombatView,
    greedy: bool,
    device: torch.device = torch.device("cpu"),
) -> tuple[Action, torch.Tensor, torch.Tensor]:
    encoding = encode_combat_view(combat_view, device)
    valid_action_mask = get_valid_action_mask(combat_view)

    probs, value = model(
        encoding, torch.tensor(valid_action_mask, dtype=torch.bool, device=device)
    )
    if greedy:
        action_idx = torch.argmax(probs)

    else:
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()

    action = action_idx_to_action(action_idx.item(), combat_view)

    return action
