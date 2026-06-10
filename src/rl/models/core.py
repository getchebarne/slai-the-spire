from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.nn as nn

from src.rl.constants import MAX_EVENT_OPTIONS
from src.rl.constants import MAX_MONSTERS
from src.rl.constants import MAX_POTION_REWARDS
from src.rl.constants import MAX_POTION_SLOTS
from src.rl.constants import MAX_RELIC_REWARDS
from src.rl.constants import MAX_RELICS
from src.rl.constants import MAX_SHOP_CARDS
from src.rl.constants import MAX_SHOP_POTIONS
from src.rl.constants import MAX_SHOP_RELICS
from src.rl.constants import MAX_SIZE_COMBAT_CARD_REWARD
from src.rl.constants import MAX_SIZE_DECK
from src.rl.constants import MAX_SIZE_DISC_PILE
from src.rl.constants import MAX_SIZE_DISCOVER
from src.rl.constants import MAX_SIZE_DRAW_PILE
from src.rl.constants import MAX_SIZE_EXHAUST
from src.rl.constants import MAX_SIZE_HAND
from src.rl.encoding.energy import _ENCODING_DIM_ENERGY
from src.rl.encoding.event import _ENCODING_DIM_EVENT_META
from src.rl.encoding.map_ import ENCODING_DIM_MAP_META
from src.rl.encoding.reward import _ENCODING_DIM_REWARD_META
from src.rl.encoding.screen import _ENCODING_DIM_SCREEN
from src.rl.encoding.shop import _DIM_SHOP_META
from src.rl.types import TGameState
from src.rl.models.entity_projector import EntityProjector
from src.rl.models.entity_transformer import EntityTransformer
from src.rl.models.map_encoder import MapEncoder


class Group(IntEnum):
    """Token group, also the per-group type-embedding index."""

    HAND = 0
    MONSTERS = 1
    CHARACTER = 2
    DISCOVER = 3
    RELICS = 4
    DECK = 5
    REWARD_CARDS = 6
    REWARD_RELIC = 7
    REWARD_POTION = 8
    SHOP_CARDS = 9
    SHOP_RELICS = 10
    SHOP_POTIONS = 11
    POTIONS = 12
    EVENT_OPTIONS = 13
    DRAW = 14
    DISCARD = 15
    EXHAUST = 16


# Fixed token order: drives the transformer cat/split and the type-index buffer
ENTITY_LAYOUT: tuple[tuple[Group, int], ...] = (
    (Group.HAND, MAX_SIZE_HAND),
    (Group.MONSTERS, MAX_MONSTERS),
    (Group.CHARACTER, 1),
    (Group.DISCOVER, MAX_SIZE_DISCOVER),
    (Group.RELICS, MAX_RELICS),
    (Group.DECK, MAX_SIZE_DECK),
    (Group.REWARD_CARDS, MAX_SIZE_COMBAT_CARD_REWARD),
    (Group.REWARD_RELIC, MAX_RELIC_REWARDS),
    (Group.REWARD_POTION, MAX_POTION_REWARDS),
    (Group.SHOP_CARDS, MAX_SHOP_CARDS),
    (Group.SHOP_RELICS, MAX_SHOP_RELICS),
    (Group.SHOP_POTIONS, MAX_SHOP_POTIONS),
    (Group.POTIONS, MAX_POTION_SLOTS),
    (Group.EVENT_OPTIONS, MAX_EVENT_OPTIONS),
    (Group.DRAW, MAX_SIZE_DRAW_PILE),
    (Group.DISCARD, MAX_SIZE_DISC_PILE),
    (Group.EXHAUST, MAX_SIZE_EXHAUST),
)

_NUM_TOKENS = sum(size for _, size in ENTITY_LAYOUT) + 1  # + the learned global token

# Packing buckets for the transformer's token dim: measured occupancy is ~17 valid
# tokens of _NUM_TOKENS=199 (p90 ~21), so the transformer runs on a compacted prefix
# padded up to the smallest covering bucket (few distinct shapes keeps kernels and
# torch.compile happy); the last bucket is the unpacked width, so packing never
# truncates.
_PACK_BUCKETS = (32, 40, 48, 56, 64, 80, 96, 112, 128, _NUM_TOKENS)


@dataclass
class CoreOutput:
    """Per-entity embeddings + global context for the action/value heads.

    Every selectable/context entity is refined by the single entity transformer.
    The shop piles carry their per-item price concatenated. Context-only groups
    (relics, reward relic/potion, draw/discard/exhaust) reach x_global via
    attention into the learned global token and are not surfaced.
    """

    x_global: torch.Tensor          # (B, dim_global)
    x_screen: torch.Tensor          # (B, _ENCODING_DIM_SCREEN) raw flats — L1 GLU context
    x_map: torch.Tensor             # (B, MAP_WIDTH, dim_map) — per-column embeddings
    x_hand: torch.Tensor            # (B, MAX_SIZE_HAND, dim_entity)
    x_monsters: torch.Tensor        # (B, MAX_MONSTERS, dim_entity)
    x_discover: torch.Tensor        # (B, MAX_SIZE_DISCOVER, dim_entity)
    x_deck: torch.Tensor            # (B, MAX_SIZE_DECK, dim_entity)
    x_reward_cards: torch.Tensor    # (B, MAX_SIZE_COMBAT_CARD_REWARD, dim_entity)
    x_shop_cards: torch.Tensor      # (B, MAX_SHOP_CARDS, dim_entity + ENCODING_DIM_PRICE)
    x_shop_relics: torch.Tensor     # (B, MAX_SHOP_RELICS, dim_entity + ENCODING_DIM_PRICE)
    x_shop_potions: torch.Tensor    # (B, MAX_SHOP_POTIONS, dim_entity + ENCODING_DIM_PRICE)
    x_potions: torch.Tensor         # (B, MAX_POTION_SLOTS, dim_entity)
    x_event_options: torch.Tensor   # (B, MAX_EVENT_OPTIONS, dim_entity)


class Core(nn.Module):
    """Shared encoder: game state -> per-entity embeddings + global context.

    One entity transformer over all 17 entity groups (ENTITY_LAYOUT order) plus a
    learned global token, padding-masked; the map has its own per-column CNN. The
    global context combines the refined global token (attention-aggregated
    entities) with per-group counts and the raw flat blocks.
    """

    def __init__(
        self,
        dim_entity: int,
        dim_global: int,
        transformer_dim_ff: int,
        transformer_num_heads: int,
        transformer_num_blocks: int,
        map_encoder_kernel_size: int,
        map_encoder_dim: int,
    ):
        super().__init__()

        self._dim_entity = dim_entity
        self._dim_global = dim_global
        self._map_encoder_dim = map_encoder_dim

        self._entity_projector = EntityProjector(dim_entity)
        self._entity_transformer = EntityTransformer(
            dim_entity, transformer_dim_ff, transformer_num_heads, transformer_num_blocks
        )
        self.last_pack_width = _NUM_TOKENS  # observability: bucket chosen by the last forward
        self._type_emb = nn.Embedding(len(Group), dim_entity)
        self._map_encoder = MapEncoder(map_encoder_kernel_size, map_encoder_dim)

        # Learned global token, refined by the transformer alongside the entities —
        # replaces per-group mean/max pooling as the entity -> global pathway.
        self._global_token = nn.Parameter(torch.empty(1, 1, dim_entity))
        nn.init.normal_(self._global_token, std=0.02)

        # Global context = refined global token + character + map summary
        # + per-group counts + raw flats.
        global_input_dim = (
            dim_entity                    # global token (attention-aggregated entities)
            + dim_entity                  # character (refined singleton)
            + map_encoder_dim             # map CNN (column-mean summary)
            + len(ENTITY_LAYOUT)          # per-group counts (mask.sum / size)
            + _ENCODING_DIM_ENERGY        # energy (raw)
            + _ENCODING_DIM_SCREEN        # screen state
            + ENCODING_DIM_MAP_META       # floor depth + act-boss + next-row kinds
            + _ENCODING_DIM_REWARD_META
            + _DIM_SHOP_META
            + _ENCODING_DIM_EVENT_META
        )
        self._global_projection = nn.Sequential(
            nn.Linear(global_input_dim, dim_global),
            nn.ReLU(),
            nn.Linear(dim_global, dim_global),
        )

        # token -> group type-embedding index (1, N), expanded to (B, N) in forward
        type_idx = torch.cat([torch.full((1, size), int(g)) for g, size in ENTITY_LAYOUT], dim=1)
        self.register_buffer("_type_idx", type_idx)

    @property
    def dim_map(self) -> int:
        return self._map_encoder_dim

    @property
    def dim_global(self) -> int:
        return self._dim_global

    def forward(self, x: TGameState) -> CoreOutput:
        p = self._entity_projector(x)
        b = x.batch_size[0]
        device = x.character.device
        char_valid = torch.ones(b, 1, dtype=torch.bool, device=device)

        # ---- Single entity transformer (cat in ENTITY_LAYOUT order) ----
        tokens = torch.cat(
            [
                p.hand.x,
                p.monsters.x,
                torch.unsqueeze(p.character, 1),
                p.discover.x,
                p.relics.x,
                p.deck.x,
                p.reward_cards.x,
                p.reward_relic.x,
                p.reward_potion.x,
                p.shop_cards.x,
                p.shop_relics.x,
                p.shop_potions.x,
                p.potions.x,
                p.event_options.x,
                p.draw.x,
                p.discard.x,
                p.exhaust.x,
            ],
            dim=1,
        )
        valid = torch.cat(
            [
                p.hand.mask,
                p.monsters.mask,
                char_valid,
                p.discover.mask,
                p.relics.mask,
                p.deck.mask,
                p.reward_cards.mask,
                p.reward_relic.mask,
                p.reward_potion.mask,
                p.shop_cards.mask,
                p.shop_relics.mask,
                p.shop_potions.mask,
                p.potions.mask,
                p.event_options.mask,
                p.draw.mask,
                p.discard.mask,
                p.exhaust.mask,
            ],
            dim=1,
        )
        tokens = tokens + self._type_emb(self._type_idx.expand(b, -1))

        # Learned global token appended after the type embeddings (its parameter
        # plays that role); always valid.
        tokens = torch.cat([tokens, self._global_token.expand(b, -1, -1)], dim=1)
        valid = torch.cat([valid, char_valid], dim=1)

        # ---- Token packing: run the transformer on a compacted prefix ----
        # Exact: masked keys contribute nothing to valid rows, and pad-slot outputs
        # are only ever read behind selection masks (subsets of `valid`), so
        # replacing them with zeros changes no logit, value, or log-prob.
        order = torch.argsort(~valid, dim=1, stable=True)  # valid tokens first
        n_valid = int(valid.sum(dim=1).max())
        s_pack = next(s for s in _PACK_BUCKETS if s >= n_valid)
        self.last_pack_width = s_pack
        pack_idx = order[:, :s_pack].unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        packed_tokens = tokens.gather(1, pack_idx)
        packed_valid = valid.gather(1, order[:, :s_pack])

        refined_packed = self._entity_transformer(packed_tokens, ~packed_valid)

        refined = torch.zeros_like(tokens).scatter(1, pack_idx, refined_packed)

        (
            x_hand,
            x_monsters,
            x_character,
            x_discover,
            x_relics,
            x_deck,
            x_reward_cards,
            x_reward_relic,
            x_reward_potion,
            x_shop_cards,
            x_shop_relics,
            x_shop_potions,
            x_potions,
            x_event_options,
            x_draw,
            x_discard,
            x_exhaust,
            x_global_token,
        ) = torch.split(refined, [*(size for _, size in ENTITY_LAYOUT), 1], dim=1)
        x_character = torch.squeeze(x_character, 1)
        x_global_token = torch.squeeze(x_global_token, 1)

        # Per-item price concatenated onto the refined shop embeddings (for buy heads)
        x_shop_cards = torch.cat([x_shop_cards, x.shop.card_prices], dim=-1)
        x_shop_relics = torch.cat([x_shop_relics, x.shop.relic_prices], dim=-1)
        x_shop_potions = torch.cat([x_shop_potions, x.shop.potion_prices], dim=-1)

        # ---- Map: per-column embeddings (B, MAP_WIDTH, dim_map) ----
        x_map = self._map_encoder(x.map_grid)

        # ---- Global context ----
        # The global token carries entity content via attention; counts carry the
        # cardinalities (deck size, pile sizes, ...) that attention/pooling blur.
        group_masks = [
            p.hand.mask,
            p.monsters.mask,
            char_valid,
            p.discover.mask,
            p.relics.mask,
            p.deck.mask,
            p.reward_cards.mask,
            p.reward_relic.mask,
            p.reward_potion.mask,
            p.shop_cards.mask,
            p.shop_relics.mask,
            p.shop_potions.mask,
            p.potions.mask,
            p.event_options.mask,
            p.draw.mask,
            p.discard.mask,
            p.exhaust.mask,
        ]
        counts = torch.cat(
            [
                mask.sum(dim=1, keepdim=True).float() / size
                for mask, (_, size) in zip(group_masks, ENTITY_LAYOUT, strict=True)
            ],
            dim=1,
        )

        x_global = self._global_projection(
            torch.cat(
                [
                    x_global_token,
                    x_character,
                    x_map.mean(dim=1),
                    counts,
                    x.combat.energy,
                    x.screen,
                    x.map_meta,
                    x.reward.meta,
                    x.shop.meta,
                    x.event.meta,
                ],
                dim=1,
            )
        )

        return CoreOutput(
            x_global=x_global,
            x_screen=x.screen,
            x_map=x_map,
            x_hand=x_hand,
            x_monsters=x_monsters,
            x_discover=x_discover,
            x_deck=x_deck,
            x_reward_cards=x_reward_cards,
            x_shop_cards=x_shop_cards,
            x_shop_relics=x_shop_relics,
            x_shop_potions=x_shop_potions,
            x_potions=x_potions,
            x_event_options=x_event_options,
        )
