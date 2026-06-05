import torch
from slai import GameState
from tensordict import tensorclass

from src.rl.constants import MAX_SIZE_DECK
from src.rl.constants import MAX_SIZE_DISC_PILE
from src.rl.constants import MAX_SIZE_DISCOVER
from src.rl.constants import MAX_SIZE_DRAW_PILE
from src.rl.constants import MAX_SIZE_HAND
from src.rl.encoding.card import encode_batch_cards
from src.rl.encoding.character import encode_batch_character
from src.rl.encoding.energy import encode_batch_energy
from src.rl.encoding.event import encode_batch_events
from src.rl.encoding.map_ import encode_batch_map
from src.rl.encoding.map_ import encode_batch_map_meta
from src.rl.encoding.monster import encode_batch_monsters
from src.rl.encoding.potion import encode_batch_potions
from src.rl.encoding.relic import encode_batch_relics
from src.rl.encoding.reward import encode_batch_rewards
from src.rl.encoding.screen import encode_batch_screen
from src.rl.encoding.shop import encode_batch_shop


@tensorclass
class TensorPadded:
    x: torch.Tensor       # (B, S, D) per-item features
    mask: torch.Tensor    # (B, S) True = valid, False = padding


@tensorclass
class TensorCombat:
    hand: TensorPadded
    draw: TensorPadded
    discard: TensorPadded
    deck: TensorPadded
    monsters: TensorPadded
    energy: torch.Tensor
    discover: TensorPadded


@tensorclass
class TensorReward:
    cards: TensorPadded
    meta: torch.Tensor


@tensorclass
class TensorShop:
    cards: TensorPadded
    relics: TensorPadded
    potions: TensorPadded
    meta: torch.Tensor


@tensorclass
class TensorEvent:
    meta: torch.Tensor
    options: TensorPadded


@tensorclass
class TensorGameState:
    # Persistent / cross-screen
    character: torch.Tensor
    relics: TensorPadded
    potions: TensorPadded
    map: torch.Tensor
    map_meta: torch.Tensor
    screen: torch.Tensor
    # Per-screen
    combat: TensorCombat
    reward: TensorReward
    shop: TensorShop
    event: TensorEvent


def encode_batch_game_state(
    batch_game_state: list[GameState], device: torch.device
) -> TensorGameState:
    # tensorclass batch dim shared by every field
    batch_size = [len(batch_game_state)]

    # Gather data from the batch of game states into separate lists
    batch_hand = []
    batch_draw = []
    batch_disc = []
    batch_deck = []
    batch_discover = []
    batch_monsters = []
    batch_character = []
    batch_energy = []
    batch_map = []
    batch_reward = []
    batch_shop = []
    batch_event = []
    batch_relics = []
    batch_potions = []
    batch_health = []
    batch_block = []
    batch_gold = []
    for game_state in batch_game_state:
        batch_hand.append(game_state.hand)
        batch_draw.append(game_state.pile_draw)
        batch_disc.append(game_state.pile_discard)
        batch_deck.append(game_state.deck)
        batch_discover.append(game_state.discover)
        batch_monsters.append(game_state.monsters)
        batch_character.append(game_state.character)
        batch_energy.append(game_state.energy)
        batch_map.append(game_state.map)
        batch_reward.append(game_state.reward)
        batch_shop.append(game_state.shop)
        batch_event.append(game_state.event)
        batch_relics.append(game_state.relics)
        batch_potions.append(game_state.potions)
        batch_health.append(game_state.character.health)
        batch_block.append(game_state.character.block)
        batch_gold.append(game_state.character.gold)

    # Monsters
    x_monsters, monsters_mask, incoming_damages = encode_batch_monsters(
        batch_monsters, batch_health, batch_block, device
    )

    # Reward (offered cards + gold/relic/potion meta)
    reward_cards, reward_cards_mask, reward_meta = encode_batch_rewards(
        batch_reward, batch_gold, device
    )

    # Shop
    (
        shop_cards,
        shop_cards_mask,
        shop_relics,
        shop_relics_mask,
        shop_potions,
        shop_potions_mask,
        shop_meta,
    ) = encode_batch_shop(batch_shop, batch_gold, device)

    # Event
    event_meta, event_options, event_options_mask = encode_batch_events(batch_event, device)

    return TensorGameState(
        character=encode_batch_character(batch_character, incoming_damages, device),
        relics=TensorPadded(*encode_batch_relics(batch_relics, device), batch_size=batch_size),
        potions=TensorPadded(*encode_batch_potions(batch_potions, device), batch_size=batch_size),
        map=encode_batch_map(batch_map, device),
        map_meta=encode_batch_map_meta(batch_map, device),
        screen=encode_batch_screen(batch_game_state, device),
        combat=TensorCombat(
            hand=TensorPadded(*encode_batch_cards(batch_hand, MAX_SIZE_HAND, device), batch_size=batch_size),
            draw=TensorPadded(*encode_batch_cards(batch_draw, MAX_SIZE_DRAW_PILE, device), batch_size=batch_size),
            discard=TensorPadded(*encode_batch_cards(batch_disc, MAX_SIZE_DISC_PILE, device), batch_size=batch_size),
            deck=TensorPadded(*encode_batch_cards(batch_deck, MAX_SIZE_DECK, device), batch_size=batch_size),
            monsters=TensorPadded(x_monsters, monsters_mask, batch_size=batch_size),
            energy=encode_batch_energy(batch_energy, device),
            discover=TensorPadded(*encode_batch_cards(batch_discover, MAX_SIZE_DISCOVER, device), batch_size=batch_size),
            batch_size=batch_size,
        ),
        reward=TensorReward(
            cards=TensorPadded(reward_cards, reward_cards_mask, batch_size=batch_size),
            meta=reward_meta,
            batch_size=batch_size,
        ),
        shop=TensorShop(
            cards=TensorPadded(shop_cards, shop_cards_mask, batch_size=batch_size),
            relics=TensorPadded(shop_relics, shop_relics_mask, batch_size=batch_size),
            potions=TensorPadded(shop_potions, shop_potions_mask, batch_size=batch_size),
            meta=shop_meta,
            batch_size=batch_size,
        ),
        event=TensorEvent(
            meta=event_meta,
            options=TensorPadded(event_options, event_options_mask, batch_size=batch_size),
            batch_size=batch_size,
        ),
        batch_size=batch_size,
    )
