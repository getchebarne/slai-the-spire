import torch
from slai import GameState
from slai import Screen

from src.rl.constants import MAX_SIZE_DECK
from src.rl.constants import MAX_SIZE_DISC_PILE
from src.rl.constants import MAX_SIZE_DISCOVER
from src.rl.constants import MAX_SIZE_DRAW_PILE
from src.rl.constants import MAX_SIZE_EXHAUST
from src.rl.constants import MAX_SIZE_HAND
from src.rl.types import TCombat
from src.rl.types import TEvent
from src.rl.types import TGameState
from src.rl.types import TPadded
from src.rl.types import TReward
from src.rl.types import TShop
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


# Deck is the full owned deck (non-empty every screen); a valid token in every
# non-combat screen (deck-edit + shop/reward synergy + map-path planning), masked
# only in combat so card plays aren't diluted by the deck tokens.
_DECK_SCREENS = frozenset(
    {Screen.Map, Screen.Chest, Screen.RestSite, Screen.Shop, Screen.Reward, Screen.Event}
)


def encode_batch_game_state(
    batch_game_state: list[GameState], device: torch.device
) -> TGameState:
    # tensorclass batch dim shared by every field
    batch_size = [len(batch_game_state)]

    # Gather data from the batch of game states into separate lists
    batch_hand = []
    batch_draw = []
    batch_disc = []
    batch_exhaust = []
    batch_deck = []
    batch_discover = []
    batch_monsters = []
    batch_character = []
    batch_energy = []
    batch_energy_current = []
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
        # The engine draws from the END of the pile — keep the next-to-draw tail
        # when over the encoder cap (encode_batch_cards truncates the head).
        batch_draw.append(game_state.pile_draw[-MAX_SIZE_DRAW_PILE:])
        batch_disc.append(game_state.pile_discard)
        batch_exhaust.append(game_state.pile_exhaust)
        batch_deck.append(game_state.deck)
        batch_discover.append(game_state.discover)
        batch_monsters.append(game_state.monsters)
        batch_character.append(game_state.character)
        batch_energy.append(game_state.energy)
        batch_energy_current.append(game_state.energy.energy_current)
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

    # Reward (offered cards + relic + potion as pure entities, gold meta)
    (
        reward_cards,
        reward_cards_mask,
        reward_relic,
        reward_relic_mask,
        reward_potion,
        reward_potion_mask,
        reward_meta,
    ) = encode_batch_rewards(batch_reward, batch_gold, device)

    # Shop
    (
        shop_cards,
        shop_cards_mask,
        shop_card_prices,
        shop_relics,
        shop_relics_mask,
        shop_relic_prices,
        shop_potions,
        shop_potions_mask,
        shop_potion_prices,
        shop_meta,
    ) = encode_batch_shop(batch_shop, batch_gold, device)

    # Event
    event_meta, event_options, event_options_mask = encode_batch_events(batch_event, device)

    # Deck cards: only a valid token in deck-building screens (see _DECK_SCREENS)
    deck_x, deck_mask = encode_batch_cards(batch_deck, batch_energy_current, MAX_SIZE_DECK, device)
    deck_screen = torch.tensor([gs.screen in _DECK_SCREENS for gs in batch_game_state], device=device)
    deck_mask = deck_mask & deck_screen.unsqueeze(1)

    hand_x, hand_pad = encode_batch_cards(batch_hand, batch_energy_current, MAX_SIZE_HAND, device)
    potions_x, potions_pad = encode_batch_potions(batch_potions, device)

    return TGameState(
        character=encode_batch_character(batch_character, incoming_damages, batch_deck, device),
        relics=TPadded(*encode_batch_relics(batch_relics, device), batch_size=batch_size),
        potions=TPadded(potions_x, potions_pad, batch_size=batch_size),
        map_grid=encode_batch_map(batch_map, device),
        map_meta=encode_batch_map_meta(batch_map, device),
        screen=encode_batch_screen(batch_game_state, device),
        combat=TCombat(
            hand=TPadded(hand_x, hand_pad, batch_size=batch_size),
            draw=TPadded(*encode_batch_cards(batch_draw, batch_energy_current, MAX_SIZE_DRAW_PILE, device), batch_size=batch_size),
            discard=TPadded(*encode_batch_cards(batch_disc, batch_energy_current, MAX_SIZE_DISC_PILE, device), batch_size=batch_size),
            exhaust=TPadded(*encode_batch_cards(batch_exhaust, batch_energy_current, MAX_SIZE_EXHAUST, device), batch_size=batch_size),
            deck=TPadded(deck_x, deck_mask, batch_size=batch_size),
            monsters=TPadded(x_monsters, monsters_mask, batch_size=batch_size),
            energy=encode_batch_energy(batch_energy, device),
            discover=TPadded(*encode_batch_cards(batch_discover, batch_energy_current, MAX_SIZE_DISCOVER, device), batch_size=batch_size),
            batch_size=batch_size,
        ),
        reward=TReward(
            cards=TPadded(reward_cards, reward_cards_mask, batch_size=batch_size),
            relic=TPadded(reward_relic, reward_relic_mask, batch_size=batch_size),
            potion=TPadded(reward_potion, reward_potion_mask, batch_size=batch_size),
            meta=reward_meta,
            batch_size=batch_size,
        ),
        shop=TShop(
            cards=TPadded(shop_cards, shop_cards_mask, batch_size=batch_size),
            card_prices=shop_card_prices,
            relics=TPadded(shop_relics, shop_relics_mask, batch_size=batch_size),
            relic_prices=shop_relic_prices,
            potions=TPadded(shop_potions, shop_potions_mask, batch_size=batch_size),
            potion_prices=shop_potion_prices,
            meta=shop_meta,
            batch_size=batch_size,
        ),
        event=TEvent(
            meta=event_meta,
            options=TPadded(event_options, event_options_mask, batch_size=batch_size),
            batch_size=batch_size,
        ),
        batch_size=batch_size,
    )
