from dataclasses import dataclass

import torch

from src.game.view.state import ViewGameState
from src.rl.encoding.card import CardPile
from src.rl.encoding.card import encode_batch_view_cards
from src.rl.encoding.character import encode_batch_view_character
from src.rl.encoding.energy import encode_batch_view_energy
from src.rl.encoding.map_ import encode_batch_view_map
from src.rl.encoding.monster import encode_batch_view_monsters


@dataclass(frozen=True)
class XGameState:
    x_hand: torch.Tensor
    x_hand_mask_pad: torch.Tensor
    x_draw: torch.Tensor
    x_draw_mask_pad: torch.Tensor
    x_disc: torch.Tensor
    x_disc_mask_pad: torch.Tensor
    x_deck: torch.Tensor
    x_deck_mask_pad: torch.Tensor
    x_combat_reward: torch.Tensor
    x_combat_reward_mask_pad: torch.Tensor
    x_monsters: torch.Tensor
    x_monsters_mask_pad: torch.Tensor
    x_character: torch.Tensor
    x_character_mask_pad: torch.Tensor
    x_energy: torch.Tensor
    x_energy_mask_pad: torch.Tensor
    x_map: torch.Tensor


def encode_batch_view_game_state(
    batch_view_game_state: list[ViewGameState], device: torch.device
) -> XGameState:
    batch_size = len(batch_view_game_state)

    # Gather data from the batch of game states into separate lists
    batch_hand = []
    batch_draw = []
    batch_disc = []
    batch_deck = []
    batch_combat_reward = []
    batch_monsters = []
    batch_character = []
    batch_energy = []
    batch_map = []
    for view_game_state in batch_view_game_state:
        batch_hand.append(view_game_state.hand)
        batch_draw.append(view_game_state.pile_draw)
        batch_disc.append(view_game_state.pile_disc)
        batch_deck.append(view_game_state.deck)
        batch_combat_reward.append(view_game_state.reward_combat)
        batch_monsters.append(view_game_state.monsters)
        batch_character.append(view_game_state.character)
        batch_energy.append(view_game_state.energy)
        batch_map.append(view_game_state.map)

    # Cards (hand, draw pile, discard pile, deck, combat rewards)
    x_hand, x_hand_mask_pad = encode_batch_view_cards(batch_hand, CardPile.HAND, device)
    x_draw, x_draw_mask_pad = encode_batch_view_cards(batch_draw, CardPile.DRAW, device)
    x_disc, x_disc_mask_pad = encode_batch_view_cards(batch_disc, CardPile.DISC, device)
    x_deck, x_deck_mask_pad = encode_batch_view_cards(batch_deck, CardPile.DECK, device)
    x_combat_reward, x_combat_reward_mask_pad = encode_batch_view_cards(
        batch_combat_reward, CardPile.COMBAT_REWARD, device
    )

    # Monsters
    x_monsters, x_monsters_mask_pad, incoming_damages = encode_batch_view_monsters(
        batch_monsters, device
    )

    # Character
    x_character = encode_batch_view_character(batch_character, incoming_damages, device)
    x_character_mask_pad = torch.ones(batch_size, 1, dtype=torch.bool, device=device)

    # Energy
    x_energy = encode_batch_view_energy(batch_energy, device)
    x_energy_mask_pad = torch.ones(batch_size, 1, dtype=torch.bool, device=device)

    # Map
    x_map = encode_batch_view_map(batch_map, device)

    # Collect all tensors
    return XGameState(
        x_hand,
        x_hand_mask_pad,
        x_draw,
        x_draw_mask_pad,
        x_disc,
        x_disc_mask_pad,
        x_deck,
        x_deck_mask_pad,
        x_combat_reward,
        x_combat_reward_mask_pad,
        x_monsters,
        x_monsters_mask_pad,
        x_character,
        x_character_mask_pad,
        x_energy,
        x_energy_mask_pad,
        x_map,
    )
