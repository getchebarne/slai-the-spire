import torch
import torch.nn as nn

from src.game.combat.constant import MAX_HAND_SIZE
from src.game.combat.constant import MAX_MONSTERS
from src.game.combat.entities import CardName
from src.game.combat.view import CombatView
from src.game.combat.view.actor import ModifierViewType
from src.game.combat.view.card import CardView
from src.game.combat.view.character import CharacterView
from src.game.combat.view.energy import EnergyView
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
MAX_LEN_DISC_PILE = 10
MAX_LEN_DRAW_PILE = 7
PAD_METADATA = -1
MODIFIER_TYPES = [ModifierViewType.WEAK, ModifierViewType.STR]


def _encode_batch_state(batch_state: list[StateView], embedding_table_state) -> torch.Tensor:
    return embedding_table_state(
        torch.tensor([STATE_IDX[state] for state in batch_state], dtype=torch.long)
    )


def _encode_batch_energy(batch_energy: list[EnergyView]) -> torch.Tensor:
    return torch.tensor([[energy_view.current] for energy_view in batch_energy], dtype=torch.float)


def _encode_batch_character(batch_character: list[CharacterView]) -> torch.Tensor:
    # Iterate over batch samples
    _tensors = []
    for character_view in batch_character:
        # Modifiers
        modifier_stacks = [
            next(
                (
                    modifier.stacks
                    for modifier in character_view.modifiers
                    if modifier.type == modifier_type
                ),
                0,
            )
            for modifier_type in MODIFIER_TYPES
        ]

        _tensors.append(
            torch.tensor(
                [character_view.health.current, character_view.block.current, *modifier_stacks],
                dtype=torch.float,
            )
        )

    return torch.stack(_tensors)


def _encode_batch_monsters(batch_monsters: list[list[MonsterView]]) -> torch.Tensor:
    # Iterate over batch samples
    _tensors = []
    for monsters in batch_monsters:
        __tensors = []
        for monster_view in monsters:
            # Modifiers
            modifier_stacks = [
                next(
                    (
                        modifier.stacks
                        for modifier in monster_view.modifiers
                        if modifier.type == modifier_type
                    ),
                    0,
                )
                for modifier_type in MODIFIER_TYPES
            ]
            __tensors.append(
                torch.tensor(
                    [
                        monster_view.health.current,
                        monster_view.block.current,
                        # Intent
                        monster_view.intent.damage or 0,
                        monster_view.intent.instances or 0,
                        monster_view.intent.block,
                        monster_view.intent.buff,
                        # Modifiers
                        *modifier_stacks,
                    ],
                    dtype=torch.float,
                )
            )

        _tensors.append(torch.cat(__tensors))

    return torch.stack(_tensors)


def _encode_batch_hand(
    batch_hand: list[list[CardView]], embedding_table_card_name: nn.Embedding
) -> torch.Tensor:
    batch_size = len(batch_hand)

    # Initialize tensors to store CardName indexes and metadata (is_active and cost)
    tensor_idxs_card_name = torch.zeros((batch_size, MAX_HAND_SIZE), dtype=torch.long)
    tensor_metadata = PAD_METADATA * torch.ones((batch_size, MAX_HAND_SIZE, 2), dtype=torch.float)

    # Iterate over batch samples
    for idx_batch, hand in enumerate(batch_hand):
        for idx_card, card_view in enumerate(hand):
            # Card name
            tensor_idxs_card_name[idx_batch, idx_card] = CARD_NAME_IDX[card_view.name]

            # Metadata
            tensor_metadata[idx_batch, idx_card, 0] = card_view.is_active
            tensor_metadata[idx_batch, idx_card, 1] = card_view.cost

    # Calculate CardName embeddings
    tensor_embeddings_card_name = embedding_table_card_name(tensor_idxs_card_name)

    # Flatten tensors
    embeddings_card_name = torch.flatten(tensor_embeddings_card_name, start_dim=1)
    embeddings_metadata = torch.flatten(tensor_metadata, start_dim=1)

    # Concatenate along feature dimension
    return torch.cat([embeddings_card_name, embeddings_metadata], dim=1)


def _encode_batch_pile(
    batch_pile: list[set[CardView]], embedding_table_card_name: nn.Embedding, max_len: int
) -> torch.Tensor:
    batch_size = len(batch_pile)

    # Initialize tensor to store CardName indexes
    tensor_idxs_card_name = torch.zeros((batch_size, max_len), dtype=torch.long)

    # Iterate over batch samples
    for idx_batch, pile in enumerate(batch_pile):
        # Sort pile to ensure consistent encoding
        pile_sort = sorted(CARD_NAME_IDX[card.name] for card in pile)
        tensor_idxs_card_name[idx_batch, : len(pile_sort)] = torch.tensor(
            pile_sort, dtype=torch.long
        )

    # Calculate CardName embeddings
    embeddings_card_name = embedding_table_card_name(tensor_idxs_card_name)

    # Flatten and return
    return torch.flatten(embeddings_card_name, start_dim=1)


class MLP(nn.Module):
    def __init__(self, layer_sizes: list[int], bias: bool = True, activation_name: str = "ReLU"):
        super().__init__()

        self._layer_sizes = layer_sizes
        self._bias = bias
        self._activation_name = activation_name

        self._mlp = self._create()

    def _create(self) -> nn.Module:
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

        # Modules
        self._card_name_embedding = nn.Embedding(
            len(CardName) + 1, card_name_embedding_size, padding_idx=0
        )
        self._state_embedding = nn.Embedding(len(StateView), state_embedding_size)
        self._mlp = MLP(linear_sizes)
        self._last = nn.Linear(linear_sizes[-1], MAX_HAND_SIZE + MAX_MONSTERS + 1, bias=True)

    def forward(self, combat_views: list[CombatView]) -> torch.Tensor:
        # Extract batch
        states = []
        characters = []
        monsters = []
        hands = []
        energies = []
        draw_piles = []
        disc_piles = []
        for combat_view in combat_views:
            states.append(combat_view.state)
            characters.append(combat_view.character)
            monsters.append(combat_view.monsters)
            hands.append(combat_view.hand)
            energies.append(combat_view.energy)
            draw_piles.append(combat_view.draw_pile)
            disc_piles.append(combat_view.disc_pile)

        # Batch encode each component and concetenate
        x = torch.cat(
            [
                _encode_batch_state(states, self._state_embedding),
                _encode_batch_character(characters),
                _encode_batch_monsters(monsters),
                _encode_batch_hand(hands, self._card_name_embedding),
                _encode_batch_energy(energies),
                _encode_batch_pile(draw_piles, self._card_name_embedding, MAX_LEN_DRAW_PILE),
                _encode_batch_pile(disc_piles, self._card_name_embedding, MAX_LEN_DISC_PILE),
            ],
            dim=1,
        )

        # Process through MLP
        x = self._mlp(x)
        x = self._last(x)

        return x
