import torch
from slai import GameState

from src.rl.types import TGameState
from src.rl.types import TPadded
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


def encode_batch_game_state(batch_game_state: list[GameState], device: torch.device) -> TGameState:
    # tensorclass batch dim shared by every field
    batch_size = [len(batch_game_state)]

    # Inputs for the bespoke encoders (the card/relic/potion class encoders pull
    # their segments straight from the game states via the registry getters)
    batch_monsters = []
    batch_character = []
    batch_energy = []
    batch_map = []
    batch_reward = []
    batch_shop = []
    batch_event = []
    batch_deck = []
    batch_health = []
    batch_block = []
    batch_gold = []
    for game_state in batch_game_state:
        batch_monsters.append(game_state.monsters)
        batch_character.append(game_state.character)
        batch_energy.append(game_state.energy)
        batch_map.append(game_state.map)
        batch_reward.append(game_state.reward)
        batch_shop.append(game_state.shop)
        batch_event.append(game_state.event)
        batch_deck.append(game_state.deck)
        batch_health.append(game_state.character.health)
        batch_block.append(game_state.character.block)
        batch_gold.append(game_state.character.gold)

    # Monsters (emit per-state incoming damage, consumed by the character encoder)
    x_monsters, monsters_mask, incoming_damages = encode_batch_monsters(
        batch_monsters, batch_health, batch_block, device
    )

    # Reward / shop / event metas + per-item shop prices
    reward_meta = encode_batch_rewards(batch_reward, batch_gold, device)
    shop_card_prices, shop_relic_prices, shop_potion_prices, shop_meta = encode_batch_shop(
        batch_shop, batch_gold, device
    )
    event_meta, event_options, event_options_mask = encode_batch_events(batch_event, device)

    # Per-class entity tensors (segment layout per src.rl.index)
    cards_x, cards_mask = encode_batch_cards(batch_game_state, device)
    relics_x, relics_mask = encode_batch_relics(batch_game_state, device)
    potions_x, potions_mask = encode_batch_potions(batch_game_state, device)

    # Deck visibility (token only in deck-building screens; hidden in combat) is enforced
    # in encode_batch_cards, which skips the deck encode in combat -> x_pad False there.
    return TGameState(
        cards=TPadded(cards_x, cards_mask, batch_size=batch_size),
        relics=TPadded(relics_x, relics_mask, batch_size=batch_size),
        potions=TPadded(potions_x, potions_mask, batch_size=batch_size),
        monsters=TPadded(x_monsters, monsters_mask, batch_size=batch_size),
        event_options=TPadded(event_options, event_options_mask, batch_size=batch_size),
        character=encode_batch_character(batch_character, incoming_damages, batch_deck, device),
        energy=encode_batch_energy(batch_energy, device),
        map_grid=encode_batch_map(batch_map, device),
        map_meta=encode_batch_map_meta(batch_map, device),
        screen=encode_batch_screen(batch_game_state, device),
        reward_meta=reward_meta,
        shop_meta=shop_meta,
        event_meta=event_meta,
        shop_card_prices=shop_card_prices,
        shop_relic_prices=shop_relic_prices,
        shop_potion_prices=shop_potion_prices,
        batch_size=batch_size,
    )
