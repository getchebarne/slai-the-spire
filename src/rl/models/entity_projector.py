import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.encoding.card import ENCODING_DIM_CARD
from src.rl.encoding.character import ENCODING_DIM_CHARACTER
from src.rl.encoding.character import SLICE_HEALTH_BLOCK_CHARACTER
from src.rl.encoding.effect import ENCODING_DIM_EFFECT
from src.rl.encoding.event import ENCODING_DIM_EVENT_OPTION
from src.rl.encoding.layout import EFFECT_SEGMENT_PENDING
from src.rl.encoding.layout import NUM_ENTITY_TOKENS
from src.rl.encoding.layout import TOKEN_OFFSET
from src.rl.encoding.layout import EntityIdx
from src.rl.encoding.layout import get_encodings_for_idx
from src.rl.encoding.modifier import ENCODING_DIM_MODIFIER
from src.rl.encoding.monster import ENCODING_DIM_MONSTER
from src.rl.encoding.monster import SLICE_HEALTH_BLOCK_MONSTER
from src.rl.encoding.potion import ENCODING_DIM_POTION
from src.rl.encoding.relic import ENCODING_DIM_RELIC
from src.rl.types import TEffects
from src.rl.types import TEntityProjection
from src.rl.types import TGameState
from src.rl.types import TModifiers
from src.rl.types import TPadded


_DIM_HEALTH_BLOCK = SLICE_HEALTH_BLOCK_MONSTER.stop - SLICE_HEALTH_BLOCK_MONSTER.start


class EntityProjector(nn.Module):
    """Factored (AlphaStar-style) entity embedder.

    First layer: each type's weight matrix is assembled from column blocks, and
    the feature families byte-aligned across encoders — health/block and
    modifiers (monster/character) — are projected by SHARED blocks, so every
    consumer's gradients accumulate into one parameter. Composition is by
    weight-cat: each type still costs a single GEMM, bit-identical to a dense
    layer. The second layer + LayerNorm are shared by all token types and run on
    one fused row tensor. Rooms come deep-processed from the GNN, so they keep a
    1-layer lift into the same LayerNorm.

    Effects no longer live in the entity row. They arrive as their own rows
    (`TGameState.effects`), go through `_effects_mlp` as ONE batched GEMM for
    the whole minibatch, and are sum-pooled into their owning token before the
    shared ReLU — so an entity's effects share its normalization. The MLP must
    stay >=2 layers: a linear encoder would make pooling additive in the input
    factors, and `kind x target x amount` binding would not survive the sum.
    """

    def __init__(self, dim_embedding: int, dim_map: int) -> None:
        super().__init__()

        self._dim_embedding = dim_embedding

        # Shared family block (first layer). Bias-free: each type's single bias
        # lives on its private block
        self._fam_health_block = nn.Linear(_DIM_HEALTH_BLOCK, dim_embedding, bias=False)

        # Per-type private blocks (identity OHEs + type-specific scalars), sized
        # from the encoder slice exports — no literals
        self._pre_card = nn.Linear(ENCODING_DIM_CARD, dim_embedding)
        self._pre_relic = nn.Linear(ENCODING_DIM_RELIC, dim_embedding)
        self._pre_potion = nn.Linear(ENCODING_DIM_POTION, dim_embedding)
        self._pre_event_option = nn.Linear(ENCODING_DIM_EVENT_OPTION, dim_embedding)
        self._post_monster = nn.Linear(
            ENCODING_DIM_MONSTER - SLICE_HEALTH_BLOCK_MONSTER.stop, dim_embedding
        )
        self._post_character = nn.Linear(
            ENCODING_DIM_CHARACTER - SLICE_HEALTH_BLOCK_CHARACTER.stop, dim_embedding
        )

        # Per-effect embedder. >=2 layers is load-bearing (see the class
        # docstring); the bias-free tail keeps pooling from manufacturing a
        # per-effect-count offset
        self._effects_mlp = nn.Sequential(
            nn.Linear(ENCODING_DIM_EFFECT, dim_embedding),
            nn.ReLU(),
            nn.Linear(dim_embedding, dim_embedding, bias=False),
        )

        # >=2 layers for the same reason as _effects_mlp: kind x stacks
        # binding would not survive a linear sum-pool
        self._modifiers_mlp = nn.Sequential(
            nn.Linear(ENCODING_DIM_MODIFIER, dim_embedding),
            nn.ReLU(),
            nn.Linear(dim_embedding, dim_embedding, bias=False),
        )

        # Shared second layer (after ReLU, on the fused all-types row tensor)
        self._proj_shared = nn.Linear(dim_embedding, dim_embedding)
        # Rooms come deep-processed from the GNN, so a 1-layer lift
        self._proj_room = nn.Linear(dim_map, dim_embedding)

        self._norm = nn.LayerNorm(dim_embedding)

        # (row_dim, ordered (columns, block) pairs) per type. The dict aliases the
        # registered modules (Module.to() mutates parameters in place)
        self._layouts = {
            EntityIdx.CARD: (ENCODING_DIM_CARD, ((slice(0, ENCODING_DIM_CARD), self._pre_card),)),
            EntityIdx.RELIC: (
                ENCODING_DIM_RELIC,
                ((slice(0, ENCODING_DIM_RELIC), self._pre_relic),),
            ),
            EntityIdx.POTION: (
                ENCODING_DIM_POTION,
                ((slice(0, ENCODING_DIM_POTION), self._pre_potion),),
            ),
            EntityIdx.MONSTER: (
                ENCODING_DIM_MONSTER,
                (
                    (SLICE_HEALTH_BLOCK_MONSTER, self._fam_health_block),
                    (
                        slice(SLICE_HEALTH_BLOCK_MONSTER.stop, ENCODING_DIM_MONSTER),
                        self._post_monster,
                    ),
                ),
            ),
            EntityIdx.EVENT_OPTION: (
                ENCODING_DIM_EVENT_OPTION,
                ((slice(0, ENCODING_DIM_EVENT_OPTION), self._pre_event_option),),
            ),
            EntityIdx.CHARACTER: (
                ENCODING_DIM_CHARACTER,
                (
                    (SLICE_HEALTH_BLOCK_CHARACTER, self._fam_health_block),
                    (
                        slice(SLICE_HEALTH_BLOCK_CHARACTER.stop, ENCODING_DIM_CHARACTER),
                        self._post_character,
                    ),
                ),
            ),
        }
        for dim_row, blocks in self._layouts.values():
            offset = 0
            for slice_, block in blocks:
                assert slice_.start == offset and block.in_features == slice_.stop - slice_.start
                offset = slice_.stop
            assert offset == dim_row, "blocks must tile the row exactly"
            assert sum(block.bias is not None for _, block in blocks) == 1, "one bias per type"
        assert list(self._layouts) == list(EntityIdx), "layouts must cover EntityIdx in order"

    def forward(self, t_game_state: TGameState, t_rooms: TPadded) -> TEntityProjection:
        batch_size = t_game_state.cards.x.shape[0]
        device = t_game_state.cards.x.device

        # First layer: per type, gather the valid rows and run ONE GEMM against the
        # weight assembled from the family + private column blocks. Every valid row
        # gets its global token position: (b, slot) -> b * WIDTH + TOKEN_OFFSET + slot
        # First pass
        firsts, positions = [], []
        for idx in EntityIdx:
            t_padded = get_encodings_for_idx(t_game_state, idx)

            _, layers = self._layouts[idx]
            _, num_entities, dim = t_padded.x.shape

            t_idx_mask = torch.nonzero(torch.flatten(t_padded.mask), as_tuple=True)[0]
            t_valid = t_padded.x.reshape(batch_size * num_entities, dim)[t_idx_mask]

            # Extract weights and biases for each Linear layer
            t_weight = torch.cat([layer.weight for _, layer in layers], dim=1)
            t_bias = next(layer.bias for _, layer in layers if layer.bias is not None)

            #
            firsts.append(F.linear(t_valid, t_weight, t_bias))
            t_batch, t_slot = t_idx_mask // num_entities, t_idx_mask % num_entities
            positions.append(t_batch * NUM_ENTITY_TOKENS + TOKEN_OFFSET[idx] + t_slot)

        t_proj = torch.cat(firsts, dim=0)  # (B', D)
        t_position = torch.cat(positions)  # (B', 1)

        # Global token slot -> fused row rank. Invalid slots stay 0; the encoder
        # only emits owners it also masks True (fuzz-asserted, not checked here:
        # it would sync)
        rank_grid = torch.zeros(batch_size * NUM_ENTITY_TOKENS, dtype=torch.long, device=device)
        rank_grid[t_position] = torch.arange(t_position.shape[0], device=device)

        # Effects/Modifiers: ONE batched GEMM each, then sum-pool every row into its
        # owner's fused row. Deliberately before the shared ReLU/LayerNorm so an
        # entity's pooled rows share its normalization
        t_pending = self._pool_effects(t_game_state.effects, rank_grid, t_proj)
        self._pool_modifiers(t_game_state.modifiers, rank_grid, t_proj)

        # Shared second layer + norm on the fused tensor (never empty: the
        # character mask is all-True)
        t_hidden = self._norm(self._proj_shared(F.relu(t_proj)))

        # One scatter into the global entity block. Zeros buffer in the INPUT dtype:
        # pad rows stay exact zeros (downstream masks multiply, and NaN*0=NaN), and
        # the fp32 LayerNorm output lands in an fp32 buffer under bf16 autocast.
        # Explicit upcast: CUDA autocast runs LayerNorm in fp32 but CPU autocast
        # emits bf16, and index_put requires matching dtypes
        t_entities = torch.zeros(
            batch_size * NUM_ENTITY_TOKENS,
            self._dim_embedding,
            dtype=t_game_state.cards.x.dtype,
            device=device,
        )
        t_entities[t_position] = t_hidden.to(t_entities.dtype)
        t_mask = torch.zeros(batch_size * NUM_ENTITY_TOKENS, dtype=torch.bool, device=device)
        t_mask[t_position] = True

        return TEntityProjection(
            entities=TPadded(
                t_entities.reshape(batch_size, NUM_ENTITY_TOKENS, self._dim_embedding),
                t_mask.reshape(batch_size, NUM_ENTITY_TOKENS),
            ),
            rooms=_project_sparse(self._proj_room, self._norm, t_rooms, self._dim_embedding),
            pending=t_pending,
        )

    def _pool_effects(
        self, t_effects: TEffects, rank_grid: torch.Tensor, t_proj: torch.Tensor
    ) -> torch.Tensor:
        """Embed every effect row once, add it into its owner's fused row in
        place, and return the pooled halt row."""
        batch_size = t_effects.x.shape[0]
        t_pending = torch.zeros(
            batch_size, self._dim_embedding, dtype=torch.float32, device=t_effects.x.device
        )
        t_batch, t_slot = torch.nonzero(t_effects.mask, as_tuple=True)
        if not t_batch.numel():
            return t_pending

        t_hidden = self._effects_mlp(t_effects.x[t_batch, t_slot])
        t_segment = t_effects.segment[t_batch, t_slot]

        # No emptiness guards: index_add_ with a zero-length index is a no-op,
        # while `bool(t_sel.any())` would force a device sync
        t_sel = t_segment == EFFECT_SEGMENT_PENDING

        # The halt owns no entity, so it becomes a global-context input instead
        t_pending.index_add_(0, t_batch[t_sel], t_hidden[t_sel].to(torch.float32))

        t_own = ~t_sel
        t_pos = rank_grid[t_batch[t_own] * NUM_ENTITY_TOKENS + t_segment[t_own]]
        t_proj.index_add_(0, t_pos, t_hidden[t_own].to(t_proj.dtype))
        return t_pending

    def _pool_modifiers(
        self, t_modifiers: TModifiers, rank_grid: torch.Tensor, t_proj: torch.Tensor
    ) -> None:
        """Embed every Modifier row once and add it into its owner's fused row
        in place (same placement rules as _pool_effects)."""
        t_batch, t_slot = torch.nonzero(t_modifiers.mask, as_tuple=True)
        if not t_batch.numel():
            return

        t_hidden = self._modifiers_mlp(t_modifiers.x[t_batch, t_slot])
        t_segment = t_modifiers.segment[t_batch, t_slot]
        t_pos = rank_grid[t_batch * NUM_ENTITY_TOKENS + t_segment]
        t_proj.index_add_(0, t_pos, t_hidden.to(t_proj.dtype))


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
        t_out[t_row_idxs] = norm(proj(t_flat[t_row_idxs])).to(t_out.dtype)

    # Reshape to original (B, S, D)
    t_out = t_out.reshape(batch_size, num_entities, dim_embedding)
    return TPadded(t_out, t_mask)
