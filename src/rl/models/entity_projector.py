import torch
import torch.nn as nn

from src.rl.encoding.card import ENCODING_DIM_CARD
from src.rl.encoding.character import ENCODING_DIM_CHARACTER
from src.rl.encoding.event import _ENCODING_DIM_EVENT_OPTION
from src.rl.encoding.monster import ENCODING_DIM_MONSTER
from src.rl.encoding.potion import ENCODING_DIM_POTION
from src.rl.encoding.relic import ENCODING_DIM_RELIC
from src.rl.types import TEntityProjection
from src.rl.types import TGameState
from src.rl.types import TPadded


def _projection(dim_in: int, dim_embedding: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dim_in, dim_embedding),
        nn.ReLU(),
        nn.Linear(dim_embedding, dim_embedding),
    )


class EntityProjector(nn.Module):
    """Project each entity type's packed encoding to the shared embedding dim.

    The card projection is shared across every card pile, and the card/relic/potion
    projections are reused for the offered reward relic/potion and the shop items
    (those entity encodings are stored pure; price lives in separate
    x.shop.*_prices fields for the buy heads).
    """

    def __init__(self, dim_embedding: int):
        super().__init__()

        self._dim_embedding = dim_embedding
        self._projection_card = _projection(ENCODING_DIM_CARD, dim_embedding)
        self._projection_monster = _projection(ENCODING_DIM_MONSTER, dim_embedding)
        self._projection_character = _projection(ENCODING_DIM_CHARACTER, dim_embedding)
        self._projection_relic = _projection(ENCODING_DIM_RELIC, dim_embedding)
        self._projection_potion = _projection(ENCODING_DIM_POTION, dim_embedding)
        self._projection_event_option = _projection(_ENCODING_DIM_EVENT_OPTION, dim_embedding)

        self._layer_norm = nn.LayerNorm(dim_embedding)

    def forward(self, x: TGameState) -> TEntityProjection:
        batch_size = x.batch_size
        norm = self._layer_norm

        def grouped(projection, srcs: list[TPadded]) -> list[TPadded]:
            # One cat'd GEMM per shared projection, over VALID rows only (occupancy
            # is ~10% — pad-slot projections are dead compute: their outputs are
            # gathered out before attention and zeros are never read downstream).
            x = torch.cat([s.x for s in srcs], dim=1)
            mask = torch.cat([s.mask for s in srcs], dim=1)
            b, s_total, d = x.shape
            rows = mask.reshape(-1).nonzero(as_tuple=True)[0]
            out = x.new_zeros(b * s_total, self._dim_embedding)
            if rows.numel():  # a group can be entirely empty for the batch (e.g. shop in combat)
                out[rows] = norm(projection(x.reshape(b * s_total, d)[rows]))
            out = out.reshape(b, s_total, self._dim_embedding)
            splits = torch.split(out, [s.x.shape[1] for s in srcs], dim=1)
            return [TPadded(o, s.mask, batch_size=batch_size) for o, s in zip(splits, srcs)]

        c = x.combat
        hand, draw, discard, exhaust, deck, discover, reward_cards, shop_cards = grouped(
            self._projection_card,
            [c.hand, c.draw, c.discard, c.exhaust, c.deck, c.discover, x.reward.cards, x.shop.cards],
        )
        relics, reward_relic, shop_relics = grouped(
            self._projection_relic, [x.relics, x.reward.relic, x.shop.relics]
        )
        potions, reward_potion, shop_potions = grouped(
            self._projection_potion, [x.potions, x.reward.potion, x.shop.potions]
        )
        (monsters,) = grouped(self._projection_monster, [c.monsters])
        (event_options,) = grouped(self._projection_event_option, [x.event.options])

        return TEntityProjection(
            # Card piles
            hand=hand,
            draw=draw,
            discard=discard,
            exhaust=exhaust,
            deck=deck,
            discover=discover,
            # Combat actors
            monsters=monsters,
            character=norm(self._projection_character(x.character)),
            # Reward
            reward_cards=reward_cards,
            reward_relic=reward_relic,
            reward_potion=reward_potion,
            # Owned
            relics=relics,
            potions=potions,
            # Shop items
            shop_cards=shop_cards,
            shop_relics=shop_relics,
            shop_potions=shop_potions,
            # Event
            event_options=event_options,
            batch_size=batch_size,
        )
