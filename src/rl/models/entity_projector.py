import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.encoding.card import ENCODING_DIM_CARD
from src.rl.encoding.character import ENCODING_DIM_CHARACTER
from src.rl.encoding.character import SLICE_HEALTH_BLOCK_CHARACTER
from src.rl.encoding.character import SLICE_MODIFIERS_CHARACTER
from src.rl.encoding.effect import ENCODING_DIM_EFFECT
from src.rl.encoding.event import ENCODING_DIM_EVENT_OPTION
from src.rl.encoding.modifier import ENCODING_DIM_MODIFIERS
from src.rl.encoding.monster import ENCODING_DIM_MONSTER
from src.rl.encoding.monster import SLICE_HEALTH_BLOCK_MONSTER
from src.rl.encoding.monster import SLICE_MODIFIERS_MONSTER
from src.rl.encoding.potion import ENCODING_DIM_POTION
from src.rl.encoding.relic import ENCODING_DIM_RELIC
from src.rl.encoding.state import EFFECT_OWNER_OFFSET_CARD
from src.rl.encoding.state import EFFECT_OWNER_OFFSET_EVENT
from src.rl.encoding.state import EFFECT_OWNER_OFFSET_POTION
from src.rl.encoding.state import EFFECT_OWNER_OFFSET_RELIC
from src.rl.encoding.state import EFFECT_SEGMENT_PENDING
from src.rl.types import TEffects
from src.rl.types import TEntityProjection
from src.rl.types import TGameState
from src.rl.types import TPadded


_DIM_HEALTH_BLOCK = SLICE_HEALTH_BLOCK_MONSTER.stop - SLICE_HEALTH_BLOCK_MONSTER.start
assert (
    SLICE_HEALTH_BLOCK_CHARACTER.stop - SLICE_HEALTH_BLOCK_CHARACTER.start == _DIM_HEALTH_BLOCK
), "monster/character health-block widths must match (shared family weights)"


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

        # Shared family blocks (first layer). Bias-free: each type's single bias
        # lives on its private block
        self._fam_health_block = nn.Linear(_DIM_HEALTH_BLOCK, dim_embedding, bias=False)
        self._fam_modifiers = nn.Linear(ENCODING_DIM_MODIFIERS, dim_embedding, bias=False)

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

        # Shared second layer (after ReLU, on the fused all-types row tensor)
        self._proj_shared = nn.Linear(dim_embedding, dim_embedding)
        # Rooms come deep-processed from the GNN, so a 1-layer lift
        self._proj_room = nn.Linear(dim_map, dim_embedding)

        self._norm = nn.LayerNorm(dim_embedding)

        # (row_dim, ordered (columns, block) pairs) per type, in TEntityProjection
        # field order. The tuple aliases the registered modules (Module.to()
        # mutates parameters in place, so aliases stay valid)
        self._layouts = (
            (ENCODING_DIM_CARD, ((slice(0, ENCODING_DIM_CARD), self._pre_card),)),
            (ENCODING_DIM_RELIC, ((slice(0, ENCODING_DIM_RELIC), self._pre_relic),)),
            (ENCODING_DIM_POTION, ((slice(0, ENCODING_DIM_POTION), self._pre_potion),)),
            (
                ENCODING_DIM_MONSTER,
                (
                    (SLICE_MODIFIERS_MONSTER, self._fam_modifiers),
                    (SLICE_HEALTH_BLOCK_MONSTER, self._fam_health_block),
                    (
                        slice(SLICE_HEALTH_BLOCK_MONSTER.stop, ENCODING_DIM_MONSTER),
                        self._post_monster,
                    ),
                ),
            ),
            (
                ENCODING_DIM_EVENT_OPTION,
                ((slice(0, ENCODING_DIM_EVENT_OPTION), self._pre_event_option),),
            ),
            (
                ENCODING_DIM_CHARACTER,
                (
                    (SLICE_MODIFIERS_CHARACTER, self._fam_modifiers),
                    (SLICE_HEALTH_BLOCK_CHARACTER, self._fam_health_block),
                    (
                        slice(SLICE_HEALTH_BLOCK_CHARACTER.stop, ENCODING_DIM_CHARACTER),
                        self._post_character,
                    ),
                ),
            ),
        )
        for dim_row, blocks in self._layouts:
            offset = 0
            for slice_, block in blocks:
                assert slice_.start == offset and block.in_features == slice_.stop - slice_.start
                offset = slice_.stop
            assert offset == dim_row, "blocks must tile the row exactly"
            assert sum(block.bias is not None for _, block in blocks) == 1, "one bias per type"

        # Effect owners: layout index -> segment base in the encoder's owner
        # space (state.py). Monsters/character carry no effect rows
        self._effect_owner_bases = (
            (0, EFFECT_OWNER_OFFSET_CARD),
            (1, EFFECT_OWNER_OFFSET_RELIC),
            (2, EFFECT_OWNER_OFFSET_POTION),
            (4, EFFECT_OWNER_OFFSET_EVENT),
        )

    def forward(self, t_game_state: TGameState, t_rooms: TPadded) -> TEntityProjection:
        entity_inputs = (
            t_game_state.cards,
            t_game_state.relics,
            t_game_state.potions,
            t_game_state.monsters,
            t_game_state.event_options,
            t_game_state.character,
        )

        # First layer: per type, gather the valid rows and run ONE GEMM against
        # the weight assembled from the family + private column blocks
        idxs, firsts, shapes = [], [], []
        for t_padded, (_, blocks) in zip(entity_inputs, self._layouts):
            batch_size, num_entities, dim_row = t_padded.x.shape
            t_idx = torch.nonzero(torch.flatten(t_padded.mask), as_tuple=True)[0]
            t_rows = t_padded.x.reshape(batch_size * num_entities, dim_row)[t_idx]
            t_weight = torch.cat([block.weight for _, block in blocks], dim=1)
            t_bias = next(block.bias for _, block in blocks if block.bias is not None)
            firsts.append(F.linear(t_rows, t_weight, t_bias))
            idxs.append(t_idx)
            shapes.append((batch_size, num_entities))

        # Effects: ONE batched GEMM for every effect row in the minibatch, then
        # sum-pool each row into its owning token. Deliberately before the shared
        # ReLU/LayerNorm so an entity's effects share its normalization
        t_pending = self._pool_effects(t_game_state.effects, idxs, firsts, shapes)

        # Shared second layer + norm on one fused tensor (never empty: the
        # character mask is all-True). 0-row types flow through as size-0 splits
        t_hidden = self._norm(self._proj_shared(F.relu(torch.cat(firsts, dim=0))))
        t_splits = torch.split(t_hidden, [t.shape[0] for t in firsts], dim=0)

        outs = []
        for t_padded, (batch_size, num_entities), t_idx, t_h in zip(
            entity_inputs, shapes, idxs, t_splits
        ):
            # Zeros buffer in the INPUT dtype: pad rows stay exact zeros (downstream
            # masks multiply, and NaN*0=NaN), and the fp32 LayerNorm output lands in
            # an fp32 buffer under bf16 autocast — same terminal property as before
            t_out = torch.zeros(
                batch_size * num_entities,
                self._dim_embedding,
                dtype=t_padded.x.dtype,
                device=t_padded.x.device,
            )
            # Explicit upcast: CUDA autocast runs LayerNorm in fp32 but CPU
            # autocast emits bf16, and index_put requires matching dtypes
            t_out[t_idx] = t_h.to(t_out.dtype)
            outs.append(
                TPadded(
                    t_out.reshape(batch_size, num_entities, self._dim_embedding), t_padded.mask
                )
            )

        return TEntityProjection(
            cards=outs[0],
            relics=outs[1],
            potions=outs[2],
            monsters=outs[3],
            events=outs[4],
            character=outs[5],
            rooms=_project_sparse(self._proj_room, self._norm, t_rooms, self._dim_embedding),
            pending=t_pending,
        )

    def _pool_effects(
        self,
        t_effects: TEffects,
        idxs: list[torch.Tensor],
        firsts: list[torch.Tensor],
        shapes: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Embed every effect row once, add it into its owner's first-layer row
        in place, and return the pooled halt row."""
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
        # while `bool(t_sel.any())` would force a device sync per type
        for layout_idx, base in self._effect_owner_bases:
            num_entities = shapes[layout_idx][1]
            t_sel = (t_segment >= base) & (t_segment < base + num_entities)
            # Owner slot -> flat (b, s) index -> rank among this type's gathered
            # valid rows. searchsorted is exact because nonzero returns ascending
            # indices and the encoder only emits owners it also masks True (that
            # precondition is fuzz-asserted, not checked here: it would sync)
            t_flat = t_batch[t_sel] * num_entities + (t_segment[t_sel] - base)
            t_pos = torch.searchsorted(idxs[layout_idx], t_flat)
            firsts[layout_idx].index_add_(
                0, t_pos, t_hidden[t_sel].to(firsts[layout_idx].dtype)
            )

        # The halt owns no entity, so it becomes a global-context input instead
        t_sel = t_segment == EFFECT_SEGMENT_PENDING
        t_pending.index_add_(0, t_batch[t_sel], t_hidden[t_sel].to(torch.float32))
        return t_pending


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
        # .to: CPU autocast emits bf16 from LayerNorm (CUDA keeps fp32); index_put
        # requires matching dtypes, so upcast explicitly (no-op when equal)
        t_out[t_row_idxs] = norm(proj(t_flat[t_row_idxs])).to(t_out.dtype)

    # Reshape to original (B, S, D)
    t_out = t_out.reshape(batch_size, num_entities, dim_embedding)
    return TPadded(t_out, t_mask)
