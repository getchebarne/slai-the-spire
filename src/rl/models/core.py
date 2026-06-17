import torch
import torch.nn as nn

from src.rl.encoding.energy import _ENCODING_DIM_ENERGY
from src.rl.encoding.event import _ENCODING_DIM_EVENT_META
from src.rl.encoding.map_ import ENCODING_DIM_MAP_META
from src.rl.encoding.reward import _ENCODING_DIM_REWARD_META
from src.rl.encoding.screen import _ENCODING_DIM_SCREEN
from src.rl.encoding.shop import _DIM_SHOP_META
from src.rl.index import GLOBAL_SLICE
from src.rl.index import NUM_TOKENS
from src.rl.index import TOKENS
from src.rl.index import TYPE_IDX
from src.rl.index import Token
from src.rl.index import TokenKind
from src.rl.index import token_counts
from src.rl.types import TCoreOutput
from src.rl.types import TGameState
from src.rl.types import TPadded
from src.rl.models.entity_projector import EntityProjector
from src.rl.models.entity_transformer import EntityTransformer
from src.rl.models.map_encoder import MapGNN


_NUM_TOKENS = NUM_TOKENS + 1  # + the learned global token

# Packing buckets for the transformer's token dim: measured occupancy is ~17 valid
# tokens of _NUM_TOKENS=206 (p90 ~21; +7 ROOM tokens, valid only on the map), so the
# transformer runs on a compacted prefix
# padded up to the smallest covering bucket (few distinct shapes keeps kernels and
# torch.compile happy); the last bucket is the unpacked width, so packing never
# truncates.
_PACK_BUCKETS = (32, 40, 48, 56, 64, 80, 96, 112, 128, _NUM_TOKENS)


class Core(nn.Module):
    """Shared encoder: game state -> per-entity embeddings + global context.

    One entity transformer over all entity tokens (registry order per src.rl.index)
    plus a learned global token, padding-masked; the map is encoded by a GNN that emits
    the next-row rooms as the ROOM token block (refined alongside every other entity) plus
    a graph readout. The global context combines the refined global token (attention-
    aggregated entities) with per-token counts and the raw flat blocks.
    """

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
        self._map_encoder_dim = map_encoder_dim

        self._entity_projector = EntityProjector(dim_entity)
        self._entity_transformer = EntityTransformer(
            dim_entity, transformer_dim_ff, transformer_num_heads, transformer_num_blocks
        )
        self.last_pack_width = _NUM_TOKENS  # observability: bucket chosen by the last forward
        self._type_emb = nn.Embedding(len(TOKENS), dim_entity)
        self._map_gnn = MapGNN(gnn_num_layers, map_encoder_dim, dim_entity)

        # Learned global token, refined by the transformer alongside the entities —
        # replaces per-group mean/max pooling as the entity -> global pathway.
        self._global_token = nn.Parameter(torch.empty(1, 1, dim_entity))
        nn.init.normal_(self._global_token, std=0.02)

        # Global context = refined global token + character + map summary
        # + per-segment counts + raw flats.
        global_input_dim = (
            dim_entity  # global token (attention-aggregated entities)
            + dim_entity  # character (refined singleton)
            + map_encoder_dim  # map CNN (column-mean summary)
            + len(TOKENS)  # per-token counts (mask.sum / size)
            + _ENCODING_DIM_ENERGY  # energy (raw)
            + _ENCODING_DIM_SCREEN  # screen state
            + ENCODING_DIM_MAP_META  # floor depth + act-boss + next-row kinds
            + _ENCODING_DIM_REWARD_META
            + _DIM_SHOP_META
            + _ENCODING_DIM_EVENT_META
        )
        self._global_projection = nn.Sequential(
            nn.Linear(global_input_dim, dim_global),
            nn.ReLU(),
            nn.Linear(dim_global, dim_global),
        )

        # token -> segment type-embedding index (1, N), expanded to (B, N) in forward.
        # Derived from the registry -> excluded from checkpoints.
        type_idx = torch.tensor(TYPE_IDX, dtype=torch.long).unsqueeze(0)
        self.register_buffer("_type_idx", type_idx, persistent=False)

    @property
    def dim_map(self) -> int:
        return self._map_encoder_dim

    @property
    def dim_global(self) -> int:
        return self._dim_global

    def forward(self, x: TGameState) -> TCoreOutput:
        p = self._entity_projector(x)  # (B, NUM_PROJECTED_TOKENS, dim_entity), registry order
        b = x.batch_size[0]
        device = x.character.device

        # ---- Map GNN: next-row rooms as the last token block + a graph readout ----
        room_tokens, x_map_readout = self._map_gnn(x.map_grid, x.room_node_idx)
        room_valid = x.room_node_idx >= 0  # (B, MAP_WIDTH) — column has a selectable room
        x_tokens = torch.cat([p.x, room_tokens], dim=1)  # (B, NUM_TOKENS, dim_entity)
        x_mask = torch.cat([p.mask, room_valid], dim=1)  # (B, NUM_TOKENS)

        # ---- Single entity transformer over the token tensor ----
        tokens = x_tokens + self._type_emb(self._type_idx.expand(b, -1))

        # Learned global token appended after the type embeddings (its parameter
        # plays that role); always valid.
        always_valid = torch.ones(b, 1, dtype=torch.bool, device=device)
        tokens = torch.cat([tokens, self._global_token.expand(b, -1, -1)], dim=1)
        valid = torch.cat([x_mask, always_valid], dim=1)

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

        # Strip the global token back off; entity tokens keep registry positions
        x_global_token = refined[:, -1]
        refined = refined[:, :-1]
        x_character = torch.squeeze(refined[:, GLOBAL_SLICE[Token(TokenKind.CHARACTER, None)]], 1)

        # ---- Global context ----
        # The global token carries entity content via attention; counts carry the
        # cardinalities (deck size, pile sizes, ...) that attention/pooling blur. The map
        # GNN readout is the whole-graph summary (replaces the old CNN column-mean).
        x_global = self._global_projection(
            torch.cat(
                [
                    x_global_token,
                    x_character,
                    x_map_readout,
                    token_counts(x_mask),
                    x.energy,
                    x.screen,
                    x.map_meta,
                    x.reward_meta,
                    x.shop_meta,
                    x.event_meta,
                ],
                dim=1,
            )
        )

        return TCoreOutput(
            x_global=x_global,
            x_screen=x.screen,
            tokens=TPadded(refined, x_mask),
            shop_card_prices=x.shop_card_prices,
            shop_relic_prices=x.shop_relic_prices,
            shop_potion_prices=x.shop_potion_prices,
            batch_size=x.batch_size,
        )
