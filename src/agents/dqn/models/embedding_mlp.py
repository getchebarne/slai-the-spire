import torch
import torch.nn as nn

from src.game.combat.constant import MAX_HAND_SIZE
from src.game.combat.constant import MAX_MONSTERS
from src.game.combat.entities import CardName
from src.game.combat.view import CombatView
from src.game.combat.view.actor import ActorView
from src.game.combat.view.actor import ModifierViewType
from src.game.combat.view.card import CardView
from src.game.combat.view.character import CharacterView
from src.game.combat.view.energy import EnergyView
from src.game.combat.view.monster import IntentView
from src.game.combat.view.monster import MonsterView
from src.game.combat.view.state import StateView


CARD_NAME_IDX = {
    CardName.STRIKE: 1,
    CardName.DEFEND: 2,
    CardName.NEUTRALIZE: 3,
    CardName.SURVIVOR: 4,
}
STATE_IDX = {
    StateView.DEFAULT: 0,
    StateView.AWAIT_CARD_TARGET: 1,
    StateView.AWAIT_EFFECT_TARGET: 2,
}


def _encode_state(state_view: StateView, embedding_table_state: nn.Embedding) -> torch.Tensor:
    return embedding_table_state(torch.tensor(STATE_IDX[state_view]))


def _encode_energy(energy_view: EnergyView) -> torch.Tensor:
    return torch.tensor([energy_view.max, energy_view.current])


def _encode_actor(actor_view: ActorView) -> torch.Tensor:
    # TODO: impove, make more readable
    mod_encs = []
    for modifier_view_type in [ModifierViewType.WEAK, ModifierViewType.STR]:
        # Find the first matching modifier, if any
        modifier_match = next(
            (modifier for modifier in actor_view.modifiers if modifier.type == modifier_view_type),
            None,
        )

        # Encode the modifier, or a default modifier if none exists
        encoded_modifier = torch.tensor(
            [1, modifier_match.stacks] if modifier_match is not None else [0, 0]
        )
        mod_encs.append(encoded_modifier)

    return torch.concat(
        [
            torch.tensor(
                [actor_view.health.max, actor_view.health.current, actor_view.block.current]
            ),
            *mod_encs,
        ]
    )


def _encode_character(character_view: CharacterView) -> torch.Tensor:
    return _encode_actor(character_view)


def _encode_intent(intent_view: IntentView) -> torch.Tensor:
    return torch.tensor(
        [
            intent_view.damage or 0,
            intent_view.instances or 0,
            intent_view.block,
            intent_view.buff,
        ],
        dtype=torch.float,
    )


def _encode_monsters(monster_views: list[MonsterView]) -> torch.Tensor:
    # TODO: fix this it's ugly
    return torch.concat(
        [
            torch.concat(
                [
                    _encode_actor(monster_view),
                    _encode_intent(monster_view.intent),
                ]
            )
            for monster_view in monster_views
        ],
    )


def _encode_card(card_view: CardView, embedding_table_card_name: nn.Embedding) -> torch.Tensor:
    return torch.concat(
        [
            torch.tensor([card_view.is_active, card_view.cost], dtype=torch.float),
            embedding_table_card_name(torch.tensor(CARD_NAME_IDX[card_view.name])),
        ]
    )


def _encode_card_pad(embedding_table_card_name: nn.Embedding) -> torch.Tensor:
    return torch.concat(
        [
            torch.tensor([-1, -1], dtype=torch.float),
            embedding_table_card_name(torch.tensor(0)),
        ]
    )


def _encode_hand(hand: list[CardView], embedding_table_card_name: nn.Embedding) -> torch.Tensor:
    # Pad to MAX_HAND_SIZE cards
    hand_pad = hand + [None] * (MAX_HAND_SIZE - len(hand))

    return torch.flatten(
        torch.concat(
            [
                (
                    _encode_card(card, embedding_table_card_name)
                    if card is not None
                    else _encode_card_pad(embedding_table_card_name)
                )
                for card in hand_pad
            ]
        )
    )


def _encode_draw_pile(
    draw_pile: set[CardView], embedding_table_card_name: nn.Embedding
) -> torch.Tensor:
    if not draw_pile:
        return _encode_card_pad(embedding_table_card_name)

    return torch.sum(
        torch.stack([_encode_card(card, embedding_table_card_name) for card in draw_pile]),
        dim=0,
    )


def _encode_disc_pile(
    disc_pile: set[CardView], embedding_table_card_name: nn.Embedding
) -> torch.Tensor:
    if not disc_pile:
        return _encode_card_pad(embedding_table_card_name)

    return torch.sum(
        torch.stack([_encode_card(card, embedding_table_card_name) for card in disc_pile]),
        dim=0,
    )


def _encode_combat(
    combat_view: CombatView,
    embedding_table_card_name: nn.Embedding,
    embedding_table_state: nn.Embedding,
) -> torch.Tensor:
    return torch.flatten(
        torch.concat(
            [
                _encode_state(combat_view.state, embedding_table_state),
                _encode_character(combat_view.character),
                _encode_monsters(combat_view.monsters),
                _encode_hand(combat_view.hand, embedding_table_card_name),
                _encode_energy(combat_view.energy),
                _encode_draw_pile(combat_view.draw_pile, embedding_table_card_name),
                _encode_disc_pile(combat_view.disc_pile, embedding_table_card_name),
            ]
        )
    )


class MLP(nn.Module):
    def __init__(self, layer_sizes: list[int], bias: bool = True, activation_name: str = "ReLU"):
        super().__init__()

        self._layer_sizes = layer_sizes
        self._bias = bias
        self._activation_name = activation_name

        self._mlp = self._create_mlp()

    def _create_mlp(self) -> nn.Module:
        layers = []
        for i in range(len(self._layer_sizes)):
            if i == 0:
                # First layer is lazy
                layers.append(nn.LazyLinear(self._layer_sizes[i], self._bias))
            else:
                layers.append(
                    nn.Linear(self._layer_sizes[i - 1], self._layer_sizes[i], self._bias)
                )

            layers.append(getattr(nn, self._activation_name)())

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._mlp(x)


class EmbeddingMLP(nn.Module):
    def __init__(
        self, card_name_embedding_size: int, state_embedding_size: int, linear_sizes: list[int]
    ):
        super().__init__()
        self._card_name_embedding_size = card_name_embedding_size
        self._state_embedding_size = state_embedding_size
        self._linear_sizes = linear_sizes

        self._embedding_table_card_name = nn.Embedding(
            len(CardName) + 1, card_name_embedding_size, padding_idx=0
        )
        self._embedding_table_state = nn.Embedding(len(StateView), state_embedding_size)
        self._mlp = MLP(linear_sizes)
        self._last = nn.Linear(linear_sizes[-1], MAX_HAND_SIZE + MAX_MONSTERS + 1, bias=True)

    def forward(self, combat_views: list[CombatView]) -> torch.Tensor:
        x = torch.stack(
            [
                _encode_combat(
                    combat_view, self._embedding_table_card_name, self._embedding_table_state
                )
                for combat_view in combat_views
            ]
        )

        x = self._mlp(x)
        x = self._last(x)

        return x
