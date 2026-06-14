import torch
import torch.nn as nn

from src.rl.encoding.card import ENCODING_DIM_CARD
from src.rl.encoding.character import ENCODING_DIM_CHARACTER
from src.rl.encoding.event import _ENCODING_DIM_EVENT_OPTION
from src.rl.encoding.monster import ENCODING_DIM_MONSTER
from src.rl.encoding.potion import ENCODING_DIM_POTION
from src.rl.encoding.relic import ENCODING_DIM_RELIC
from src.rl.index import NUM_TOKENS
from src.rl.index import EntityClass
from src.rl.types import TGameState
from src.rl.types import TPadded


class EntityProjector(nn.Module):
    """One shared projection per entity class into the entity-embedding space.

    The encoding layer delivers each class pre-concatenated (segments per
    src.rl.index), so projection is one sparse GEMM per class — no cat/split.
    Emits the single token tensor (B, index.NUM_TOKENS, dim_embedding) with the
    class blocks in registry order.
    """

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

    def forward(self, x: TGameState) -> TPadded:
        batch_size = x.batch_size

        # Character is the only singleton entity (flat, always valid)
        character = TPadded(
            torch.unsqueeze(self._norm(self._proj_character(x.character)), 1),
            torch.ones(batch_size[0], 1, dtype=torch.bool, device=x.character.device),
        )

        class_projections: dict[EntityClass, TPadded] = {
            EntityClass.CARD: _project_sparse(
                self._proj_card, self._norm, x.cards, self._dim_embedding
            ),
            EntityClass.RELIC: _project_sparse(
                self._proj_relic, self._norm, x.relics, self._dim_embedding
            ),
            EntityClass.POTION: _project_sparse(
                self._proj_potion, self._norm, x.potions, self._dim_embedding
            ),
            EntityClass.MONSTER: _project_sparse(
                self._proj_monster, self._norm, x.monsters, self._dim_embedding
            ),
            EntityClass.EVENT: _project_sparse(
                self._proj_event_option, self._norm, x.event_options, self._dim_embedding
            ),
            EntityClass.CHARACTER: character,
        }

        # Class blocks are contiguous in registry order, so the cat lands every
        # segment at its index.GLOBAL_SLICE position
        x_out = torch.cat([class_projections[c].x for c in EntityClass], dim=1)
        mask = torch.cat([class_projections[c].mask for c in EntityClass], dim=1)
        assert x_out.shape[1] == NUM_TOKENS

        return TPadded(x_out, mask, batch_size=batch_size)


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
