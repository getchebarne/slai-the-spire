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


class EntityProjector(nn.Module):
    def __init__(self, dim_embedding: int):
        super().__init__()

        self._dim_embedding = dim_embedding

        # Projection blocks
        self._proj_card = _get_projection(ENCODING_DIM_CARD, dim_embedding)
        self._proj_monster = _get_projection(ENCODING_DIM_MONSTER, dim_embedding)
        self._proj_character = _get_projection(ENCODING_DIM_CHARACTER, dim_embedding)
        self._proj_relic = _get_projection(ENCODING_DIM_RELIC, dim_embedding)
        self._proj_potion = _get_projection(ENCODING_DIM_POTION, dim_embedding)
        self._proj_event_option = _get_projection(_ENCODING_DIM_EVENT_OPTION, dim_embedding)

        self._norm = nn.LayerNorm(dim_embedding)

    def forward(self, x: TGameState) -> TEntityProjection:
        batch_size = x.batch_size

        hand, draw, discard, exhaust, deck, discover, reward_cards, shop_cards = _forward_grouped(
            self._proj_card,
            self._norm,
            [
                x.combat.hand,
                x.combat.draw,
                x.combat.discard,
                x.combat.exhaust,
                x.combat.deck,
                x.combat.discover,
                x.reward.cards,
                x.shop.cards,
            ],
            self._dim_embedding,
        )
        relics, reward_relic, shop_relics = _forward_grouped(
            self._proj_relic,
            self._norm,
            [x.relics, x.reward.relic, x.shop.relics],
            self._dim_embedding,
        )
        potions, reward_potion, shop_potions = _forward_grouped(
            self._proj_potion,
            self._norm,
            [x.potions, x.reward.potion, x.shop.potions],
            self._dim_embedding,
        )
        monsters = _project_sparse(
            self._proj_monster, self._norm, x.combat.monsters, self._dim_embedding
        )
        event_options = _project_sparse(
            self._proj_event_option, self._norm, x.event.options, self._dim_embedding
        )

        # Character is the only singleton entity
        character = self._norm(self._proj_character(x.character))

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
            character=character,
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
            # TODO: add `all`, contained all concatenated entities and their masks
        )


def _get_projection(dim_in: int, dim_embedding: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dim_in, dim_embedding),
        nn.ReLU(),
        nn.Linear(dim_embedding, dim_embedding),
    )


# Project only the valid (non-pad) rows of a padded entity tensor
def _project_sparse(
    proj: nn.Module,
    norm: nn.Module,
    tensor_padded: TPadded,
    dim_embedding: int,
) -> TPadded:
    x = tensor_padded.x
    mask = tensor_padded.mask

    # Initialize empty result tensor w/ one row per batch-entity
    batch_size, num_entities, dim_entity = x.shape
    x_out = torch.zeros(batch_size * num_entities, dim_embedding, dtype=x.dtype)

    # Compute which rows actually need to be projected
    row_idxs = torch.nonzero(torch.flatten(mask), as_tuple=True)[0]

    # Compute projection
    if row_idxs.numel():
        x_out[row_idxs] = norm(proj(x.reshape(batch_size * num_entities, dim_entity)[row_idxs]))

    # Reshape to original (B, S, D)
    x_out = x_out.reshape(batch_size, num_entities, dim_embedding)
    return TPadded(x_out, mask)


# TODO: also return concatenated `x_out`. Return tensor of all projected entities
def _forward_grouped(
    proj: nn.Module,
    norm: nn.Module,
    tensors_padded: list[TPadded],
    dim_embedding: int,
) -> list[TPadded]:
    # Concatenate all tensors and their masks preserving order
    xs = []
    masks = []
    for tensor_padded in tensors_padded:
        xs.append(tensor_padded.x)
        masks.append(tensor_padded.mask)

    x_cat = torch.cat(xs, dim=1)
    mask_cat = torch.cat(masks, dim=1)

    # Project valid rows over the concatenated group
    x_out = _project_sparse(proj, norm, TPadded(x_cat, mask_cat), dim_embedding).x

    # Split back into the original per-source sequences
    x_out_splits = torch.split(
        x_out, [tensor_padded.x.shape[1] for tensor_padded in tensors_padded], dim=1
    )
    return [
        TPadded(x_out_split, tensor_padded.mask)
        for x_out_split, tensor_padded in zip(x_out_splits, tensors_padded)
    ]
