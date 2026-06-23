import torch
import torch.nn as nn

from src.rl.encoding.card import ENCODING_DIM_CARD
from src.rl.encoding.character import ENCODING_DIM_CHARACTER
from src.rl.encoding.event import ENCODING_DIM_EVENT_OPTION
from src.rl.encoding.monster import ENCODING_DIM_MONSTER
from src.rl.encoding.potion import ENCODING_DIM_POTION
from src.rl.encoding.relic import ENCODING_DIM_RELIC
from src.rl.types import TEntityProjection
from src.rl.types import TGameState
from src.rl.types import TPadded


class EntityProjector(nn.Module):
    """One shared projection per entity type; returns the per-type blocks (no cat/split)."""

    def __init__(self, dim_embedding: int, dim_map: int):
        super().__init__()

        self._dim_embedding = dim_embedding

        # Projection blocks
        self._proj_card = _get_projection(ENCODING_DIM_CARD, dim_embedding)
        self._proj_monster = _get_projection(ENCODING_DIM_MONSTER, dim_embedding)
        self._proj_character = _get_projection(ENCODING_DIM_CHARACTER, dim_embedding)
        self._proj_relic = _get_projection(ENCODING_DIM_RELIC, dim_embedding)
        self._proj_potion = _get_projection(ENCODING_DIM_POTION, dim_embedding)
        self._proj_event_option = _get_projection(ENCODING_DIM_EVENT_OPTION, dim_embedding)
        # Rooms come deep-processed from the GNN, so a 1-layer lift (the others need a 2-layer MLP)
        self._proj_room = nn.Linear(dim_map, dim_embedding)

        self._norm = nn.LayerNorm(dim_embedding)

    def forward(self, t_game_state: TGameState, t_rooms: TPadded) -> TEntityProjection:
        return TEntityProjection(
            cards=_project_sparse(
                self._proj_card, self._norm, t_game_state.cards, self._dim_embedding
            ),
            relics=_project_sparse(
                self._proj_relic, self._norm, t_game_state.relics, self._dim_embedding
            ),
            potions=_project_sparse(
                self._proj_potion, self._norm, t_game_state.potions, self._dim_embedding
            ),
            monsters=_project_sparse(
                self._proj_monster, self._norm, t_game_state.monsters, self._dim_embedding
            ),
            events=_project_sparse(
                self._proj_event_option, self._norm, t_game_state.event_options, self._dim_embedding
            ),
            character=_project_sparse(
                self._proj_character, self._norm, t_game_state.character, self._dim_embedding
            ),
            rooms=_project_sparse(self._proj_room, self._norm, t_rooms, self._dim_embedding),
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
    t_padded: TPadded,
    dim_embedding: int,
) -> TPadded:
    t_x = t_padded.x
    t_mask = t_padded.mask

    # Initialize empty result tensor w/ one row per batch-entity
    batch_size, num_entities, dim_entity = t_x.shape
    t_out = torch.zeros(
        batch_size * num_entities, dim_embedding, dtype=t_x.dtype, device=t_x.device
    )

    # Compute which rows actually need to be projected
    t_row_idxs = torch.nonzero(torch.flatten(t_mask), as_tuple=True)[0]

    # Compute projection
    if t_row_idxs.numel():
        t_flat = t_x.reshape(batch_size * num_entities, dim_entity)
        t_out[t_row_idxs] = norm(proj(t_flat[t_row_idxs]))

    # Reshape to original (B, S, D)
    t_out = t_out.reshape(batch_size, num_entities, dim_embedding)
    return TPadded(t_out, t_mask)
