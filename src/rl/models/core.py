import torch
import torch.nn as nn

from src.rl.encoding.card import SLICE_CARDS
from src.rl.encoding.character import SLICE_CHARACTER
from src.rl.encoding.energy import ENCODING_DIM_ENERGY
from src.rl.encoding.event import ENCODING_DIM_EVENT_META
from src.rl.encoding.event import SLICE_EVENTS
from src.rl.encoding.map_ import ENCODING_DIM_MAP_META
from src.rl.encoding.map_ import SLICE_ROOMS
from src.rl.encoding.monster import SLICE_MONSTERS
from src.rl.encoding.potion import SLICE_POTIONS
from src.rl.encoding.relic import SLICE_RELICS
from src.rl.encoding.reward import ENCODING_DIM_REWARD_META
from src.rl.encoding.screen import ENCODING_DIM_SCREEN
from src.rl.encoding.shop import ENCODING_DIM_PRICE
from src.rl.models.entity_projector import EntityProjector
from src.rl.models.entity_transformer import EntityTransformer
from src.rl.models.map_encoder import MapGNN
from src.rl.types import ACTION_TYPE_POOL
from src.rl.types import SliceKind
from src.rl.types import TCoreOutput
from src.rl.types import TEntityProjection
from src.rl.types import TGameState
from src.rl.types import TPadded


# Cat order = global token order; ROOM last (GNN node features, projected with the rest).
SLICE_ALL = (
    SLICE_CARDS
    + SLICE_RELICS
    + SLICE_POTIONS
    + SLICE_MONSTERS
    + SLICE_EVENTS
    + SLICE_CHARACTER
    + SLICE_ROOMS
)


def _build_token_layout() -> tuple[dict[SliceKind, slice], int]:
    offsets = {}
    offset = 0
    for slice_ in SLICE_ALL:
        offsets[slice_.kind] = slice(offset, offset + slice_.size)
        offset += slice_.size

    return offsets, offset


_SLICE_OFFSETS, _NUM_TOKENS = _build_token_layout()
_SELECTABLE = set(ACTION_TYPE_POOL.values()) | {SliceKind.MONSTERS}


class Core(nn.Module):
    def __init__(
        self,
        dim_entity: int,
        dim_global: int,
        transformer_dim_ff: int,
        transformer_num_heads: int,
        transformer_num_blocks: int,
        gnn_num_layers: int,
        map_encoder_dim: int,
    ):
        super().__init__()

        self._dim_entity = dim_entity
        self._dim_global = dim_global

        self._entity_projector = EntityProjector(dim_entity, map_encoder_dim)
        self._entity_transformer = EntityTransformer(
            dim_entity, transformer_dim_ff, transformer_num_heads, transformer_num_blocks
        )
        self._type_emb = nn.Embedding(len(SLICE_ALL), dim_entity)
        self._map_gnn = MapGNN(gnn_num_layers, map_encoder_dim)

        # Learned global token, refined with the entities (replaces pooling).
        self._entity_global = nn.Parameter(torch.empty(1, 1, dim_entity))
        nn.init.normal_(self._entity_global, std=0.02)

        # Global context: global token + character + map readout + per-slice counts + raw flats.
        global_input_dim = (
            dim_entity  # global token (attention-aggregated entities)
            + dim_entity  # character (refined singleton)
            + map_encoder_dim  # map GNN readout (whole-graph summary)
            + len(SLICE_ALL)
            - 1  # per-slice counts (mask.sum / size); CHARACTER excluded (const)
            + ENCODING_DIM_ENERGY  # energy (raw)
            + ENCODING_DIM_SCREEN  # screen state
            + ENCODING_DIM_MAP_META  # floor depth + act-boss + next-row kinds
            + ENCODING_DIM_REWARD_META
            + ENCODING_DIM_PRICE  # shop purge service price
            + ENCODING_DIM_EVENT_META
        )
        self._global_projection = nn.Sequential(
            nn.Linear(global_input_dim, dim_global),
            nn.ReLU(),
            nn.Linear(dim_global, dim_global),
        )

        # token slot -> SLICE_ALL index; derived from layout, excluded from checkpoints.
        t_type_idx = torch.tensor(
            [i for i, s in enumerate(SLICE_ALL) for _ in range(s.size)], dtype=torch.long
        ).unsqueeze(0)
        self.register_buffer("_type_idx", t_type_idx, persistent=False)

    @property
    def dim_global(self) -> int:
        return self._dim_global

    def forward(self, t_game_state: TGameState) -> TCoreOutput:
        # Map GNN: next-row room features + whole-graph readout
        t_room_features, t_map_readout = self._map_gnn(
            t_game_state.map_grid, t_game_state.room_node_idx, t_game_state.room_mask
        )
        t_blocks = self._entity_projector(t_game_state, t_room_features)

        t_tokens = self._assemble_tokens(t_blocks)
        t_refined = self._refine_tokens(t_tokens)

        return TCoreOutput(
            global_=self._global_context(t_game_state, t_refined, t_map_readout),
            pool={kind: t_refined.x[:, _SLICE_OFFSETS[kind]] for kind in _SELECTABLE},
        )

    def _assemble_tokens(self, t_blocks: TEntityProjection) -> TPadded:
        batch = t_blocks.cards.x.shape[0]
        t_entities = torch.cat(
            [
                t_blocks.cards.x,
                t_blocks.relics.x,
                t_blocks.potions.x,
                t_blocks.monsters.x,
                t_blocks.events.x,
                t_blocks.character.x,
                t_blocks.rooms.x,
            ],
            dim=1,
        )
        t_entities = t_entities + self._type_emb(self._type_idx.expand(batch, -1))
        t_valid = torch.cat(
            [
                t_blocks.cards.mask,
                t_blocks.relics.mask,
                t_blocks.potions.mask,
                t_blocks.monsters.mask,
                t_blocks.events.mask,
                t_blocks.character.mask,
                t_blocks.rooms.mask,
            ],
            dim=1,
        )

        t_entity_global = self._entity_global.expand(batch, -1, -1)
        t_always_valid = torch.ones(batch, 1, dtype=torch.bool, device=t_entities.device)
        return TPadded(
            torch.cat([t_entities, t_entity_global], dim=1),
            torch.cat([t_valid, t_always_valid], dim=1),
        )

    def _refine_tokens(self, t_tokens: TPadded) -> TPadded:
        # Pack to the batch's true max token count: attention is O(width^2), and CPU-eager has no
        # recompilation penalty for a per-batch width (re-bucket if you compile or move to GPU).
        pack_width = int(t_tokens.mask.sum(dim=1).max())

        # Gather valid elements across each sample in the batch
        t_idx_keep_mask = torch.argsort(~t_tokens.mask, dim=1, stable=True, descending=False)
        t_idx_keep_mask = t_idx_keep_mask[:, :pack_width]
        t_idx_keep_x = t_idx_keep_mask.unsqueeze(-1).expand(-1, -1, t_tokens.x.shape[-1])
        t_tokens_packed = TPadded(
            t_tokens.x.gather(1, t_idx_keep_x), t_tokens.mask.gather(1, t_idx_keep_mask)
        )

        # Run transformer
        t_tokens_ref = self._entity_transformer(t_tokens_packed)

        # Scatter the refined tokens back to their original slots. Padding must be zeroed (not
        # empty_like): downstream masks multiply (x*0), and NaN*0=NaN, so uninitialized NaN-pattern
        # bytes propagate — observed crashing the x86 CPU path with NaN-pattern garbage.
        t_scattered = torch.zeros_like(t_tokens.x)
        t_scattered.scatter_(1, t_idx_keep_x, t_tokens_ref.x)
        return TPadded(t_scattered, t_tokens.mask)

    def _global_context(
        self, t_game_state: TGameState, t_refined: TPadded, t_map_readout: torch.Tensor
    ) -> torch.Tensor:
        t_entity_global = t_refined.x[:, -1]
        t_character = t_refined.x[:, _SLICE_OFFSETS[SliceKind.CHARACTER]].squeeze(1)
        return self._global_projection(
            torch.cat(
                [
                    t_entity_global,
                    t_character,
                    t_map_readout,
                    # Counts exclude the learned global token
                    _get_token_counts(t_refined.mask[:, :-1]),
                    t_game_state.energy,
                    t_game_state.screen,
                    t_game_state.map_meta,
                    t_game_state.reward_meta,
                    t_game_state.shop_meta,
                    t_game_state.event_meta,
                ],
                dim=1,
            )
        )


def _get_token_counts(t_mask: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            t_mask[:, _SLICE_OFFSETS[slice_.kind]].sum(dim=1, keepdim=True).float() / slice_.size
            for slice_ in SLICE_ALL
            if slice_.kind is not SliceKind.CHARACTER  # singleton: count is a constant 1.0
        ],
        dim=1,
    )
