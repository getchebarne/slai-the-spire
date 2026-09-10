import functools

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.rl.constants import MAP_WIDTH
from src.rl.constants import MAX_CARD_REWARD_ROLL
from src.rl.constants import MAX_COMBAT_CARD_REWARD
from src.rl.constants import MAX_EVENT_OPTIONS
from src.rl.constants import MAX_MONSTERS
from src.rl.constants import MAX_PENDING_SELECTED
from src.rl.constants import MAX_POTION_REWARDS
from src.rl.constants import MAX_POTION_SLOTS
from src.rl.constants import MAX_RELIC_REWARDS
from src.rl.constants import MAX_RELICS
from src.rl.constants import MAX_SHOP_POTIONS
from src.rl.constants import MAX_SHOP_RELICS
from src.rl.constants import MAX_SIZE_CARD_PILE_DECK
from src.rl.constants import MAX_SIZE_CARD_PILE_DISCARD
from src.rl.constants import MAX_SIZE_CARD_PILE_DISCOVER
from src.rl.constants import MAX_SIZE_CARD_PILE_DRAW
from src.rl.constants import MAX_SIZE_CARD_PILE_EVENT_ROLL
from src.rl.constants import MAX_SIZE_CARD_PILE_EXHAUST
from src.rl.constants import MAX_SIZE_CARD_PILE_HAND
from src.rl.constants import MAX_SIZE_CARD_PILE_NIGHTMARE
from src.rl.constants import MAX_SIZE_CARD_PILE_STASIS
from src.rl.constants import MAX_SIZE_CARD_PLAY_QUEUE
from src.rl.constants import MAX_SIZE_POTION_EVENT_ROLL
from src.rl.constants import MAX_SIZE_RELIC_EVENT_ROLL
from src.rl.constants import MAX_SIZE_SHOP_CARDS
from src.rl.encoding.card import SLICE_KINDS_CARD_REWARDS
from src.rl.encoding.energy import ENCODING_DIM_ENERGY
from src.rl.encoding.event import ENCODING_DIM_EVENT_META
from src.rl.encoding.focus import ENCODING_DIM_FOCUS
from src.rl.encoding.focus import OFF_FOCUS_DRAW_ORDER_VISIBLE
from src.rl.encoding.map_ import ENCODING_DIM_MAP_META
from src.rl.encoding.reward import ENCODING_DIM_REWARD_META
from src.rl.encoding.shop import ENCODING_DIM_SHOP_META
from src.rl.encoding.state import ENCODING_DIM_ASCENSION
from src.rl.models.entity_projector import EntityProjector
from src.rl.models.entity_transformer import EntityTransformer
from src.rl.models.map_encoder import MapGNN
from src.rl.types import SliceKind
from src.rl.types import X_CoreOutput
from src.rl.types import X_Entity
from src.rl.types import X_GameState


# Count features: one per kind except the character singleton (a constant 1.0),
# each normalized by its collection's cap — the slice each encoder applies to that
# pile (a count never exceeds it). Window order: cards, relics, potions, monsters,
# event options, rooms
_COUNT_WIDTHS: tuple[tuple[SliceKind, int], ...] = (
    (SliceKind.CARD_PILE_HAND, MAX_SIZE_CARD_PILE_HAND),
    (SliceKind.CARD_PILE_DISCOVER, MAX_SIZE_CARD_PILE_DISCOVER),
    (SliceKind.CARD_PENDING_SELECTED, MAX_PENDING_SELECTED),
    (SliceKind.CARD_REWARD_0, MAX_CARD_REWARD_ROLL),
    (SliceKind.CARD_REWARD_1, MAX_COMBAT_CARD_REWARD),
    (SliceKind.CARD_REWARD_2, MAX_COMBAT_CARD_REWARD),
    (SliceKind.CARD_REWARD_3, MAX_COMBAT_CARD_REWARD),
    (SliceKind.CARD_REWARD_4, MAX_COMBAT_CARD_REWARD),
    (SliceKind.CARD_SHOP, MAX_SIZE_SHOP_CARDS),
    (SliceKind.CARD_PILE_EVENT_ROLL, MAX_SIZE_CARD_PILE_EVENT_ROLL),
    (SliceKind.CARD_MATCH_FLIPPED, 1),
    (SliceKind.CARD_PILE_STASIS, MAX_SIZE_CARD_PILE_STASIS),
    (SliceKind.CARD_PLAY_QUEUE, MAX_SIZE_CARD_PLAY_QUEUE),
    (SliceKind.CARD_PILE_NIGHTMARE, MAX_SIZE_CARD_PILE_NIGHTMARE),
    (SliceKind.CARD_PILE_DRAW, MAX_SIZE_CARD_PILE_DRAW),
    (SliceKind.CARD_PILE_DISCARD, MAX_SIZE_CARD_PILE_DISCARD),
    (SliceKind.CARD_PILE_EXHAUST, MAX_SIZE_CARD_PILE_EXHAUST),
    (SliceKind.CARD_PILE_DECK, MAX_SIZE_CARD_PILE_DECK),
    (SliceKind.RELIC_REWARD, MAX_RELIC_REWARDS),
    (SliceKind.RELIC_SHOP, MAX_SHOP_RELICS),
    (SliceKind.RELIC_EVENT_ROLL, MAX_SIZE_RELIC_EVENT_ROLL),
    (SliceKind.RELIC_OWNED, MAX_RELICS),
    (SliceKind.POTION_OWNED, MAX_POTION_SLOTS),
    (SliceKind.POTION_REWARD, MAX_POTION_REWARDS),
    (SliceKind.POTION_SHOP, MAX_SHOP_POTIONS),
    (SliceKind.POTION_EVENT_ROLL, MAX_SIZE_POTION_EVENT_ROLL),
    (SliceKind.MONSTERS, MAX_MONSTERS),
    (SliceKind.EVENT_OPTIONS, MAX_EVENT_OPTIONS),
    (SliceKind.ROOMS, MAP_WIDTH),
)
assert {kind for kind, _ in _COUNT_WIDTHS} == set(SliceKind) - {
    SliceKind.CHARACTER
}, "every kind counts once"
_REWARD_LO = SliceKind.CARD_REWARD_0
_REWARD_HI = SLICE_KINDS_CARD_REWARDS[-1]


def draw_depth(x_idxs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per window token, its depth in the draw pile: the cards above it, 0 off the
    draw tokens. Per row, whether the pile shows an order: at least 2 draw tokens.

    The draw tokens are the pile's top cards in engine order, bottom first, so the
    last rank is the next card drawn: depth = count - 1 - rank."""
    x_is_draw = x_idxs == SliceKind.CARD_PILE_DRAW
    x_count = x_is_draw.sum(dim=1, keepdim=True)
    # A draw token's running count is its rank + 1
    x_depth = (x_count - x_is_draw.cumsum(dim=1)) * x_is_draw
    return x_depth, x_count.squeeze(1) >= 2


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
        checkpoint_privileged: bool,
    ) -> None:
        super().__init__()

        self._dim_entity = dim_entity
        self._dim_global = dim_global

        self._entity_projector = EntityProjector(dim_entity, map_encoder_dim)
        self._entity_transformer = EntityTransformer(
            dim_entity, transformer_dim_ff, transformer_num_heads, transformer_num_blocks
        )
        # Token identity: one row per SliceKind, looked up by the window's kind row
        self._type_emb = nn.Embedding(len(SliceKind), dim_entity)
        self._map_gnn = MapGNN(gnn_num_layers, map_encoder_dim)

        # Global context: global token + character + map readout + per-kind counts + raw flats
        global_input_dim = (
            dim_entity  # global token (attention-aggregated entities)
            + dim_entity  # character (refined singleton)
            + map_encoder_dim  # map GNN readout (whole-graph summary)
            + len(_COUNT_WIDTHS)  # per-kind token counts, sqrt-normalized by the block width
            + ENCODING_DIM_ENERGY  # energy (raw)
            + ENCODING_DIM_FOCUS  # focus row
            + ENCODING_DIM_ASCENSION  # ascension thermometer (curriculum level)
            + ENCODING_DIM_MAP_META  # floor depth + act-boss + next-row kinds
            + ENCODING_DIM_REWARD_META
            + ENCODING_DIM_SHOP_META  # shop purge price triple + purged flag
            + ENCODING_DIM_EVENT_META
            + dim_entity  # pooled pending-halt effect (EntityProjector)
        )
        # Output LN: HeadActionType/HeadValue read this vector without their
        # own norm, so scale drift here moved logits directly (EFFECT-III fix)
        self._global_projection = nn.Sequential(
            nn.Linear(global_input_dim, dim_global),
            nn.ReLU(),
            nn.Linear(dim_global, dim_global),
            nn.LayerNorm(dim_global),
        )

        # Count features: kind values to count, and each kind's normalizer (its
        # block's width); derived from the widths, excluded from checkpoints
        self.register_buffer(
            "_count_kinds",
            torch.tensor([kind for kind, _ in _COUNT_WIDTHS]).view(1, 1, -1),
            persistent=False,
        )
        self.register_buffer(
            "_count_norm",
            torch.tensor([float(width) for _, width in _COUNT_WIDTHS]).view(1, -1),
            persistent=False,
        )

        # The per-bundle kinds carry their own type rows; this shared embedding
        # is the global CARD_REWARD identity summed on top of all of them
        self._reward_emb = nn.Embedding(2, dim_entity, padding_idx=0)

        # Draw order: one row per depth, shared by both passes. The value head trains it
        # on every ordered pile (under Frozen Eye through the actor pass, the only time
        # the policy reads it)
        self._draw_depth_emb = nn.Embedding(MAX_SIZE_CARD_PILE_DRAW, dim_entity)
        # Memory switch: the privileged rows run apart from the actor's and recompute
        # in the backward instead of storing their activations
        self._checkpoint_privileged = checkpoint_privileged

    @property
    def dim_global(self) -> int:
        return self._dim_global

    def forward(self, x_game_state: X_GameState) -> X_CoreOutput:
        # Map GNN: next-row room features + whole-graph readout
        x_rooms, x_map_readout = self._map_gnn(
            x_game_state.map_grid, x_game_state.room_node_idx, x_game_state.room_idxs
        )
        # The window as one X_Entity (the six blocks then the rooms) + the pooled pending halt
        x_entities, x_pending = self._entity_projector(x_game_state, x_rooms)

        # The rest runs as two passes, both adding the draw depths to a pile that shows
        # an order: the privileged pass (the value head's) always, the actor pass (the
        # policy heads') only under Frozen Eye. They differ only on an ordered pile the
        # actor can't see, so only those rows run the privileged pass; on every other
        # row the actor pass serves the value head too
        x_depth, x_has_order = draw_depth(x_entities.x_slice_kind)
        x_visible = x_game_state.focus[:, OFF_FOCUS_DRAW_ORDER_VISIBLE] > 0
        x_ordered_actor = x_has_order & x_visible
        x_rows_actor = torch.arange(x_ordered_actor.shape[0], device=x_ordered_actor.device)
        x_rows_privileged = torch.nonzero(x_has_order & ~x_visible, as_tuple=True)[0]
        x_ordered_privileged = torch.ones_like(x_rows_privileged, dtype=torch.bool)
        # The passes share their inputs; each names its rows and which of them add depths
        run_pass = functools.partial(
            self._pass, x_game_state, x_entities, x_map_readout, x_pending, x_depth
        )

        if (
            self._checkpoint_privileged
            and torch.is_grad_enabled()
            and x_rows_privileged.shape[0] > 0
        ):
            # Checkpointing needs the privileged rows in a call of their own. Without
            # autograd (a rollout stores nothing) or without such rows, the batched call
            # serves. They run first because autograd backpropagates the later pass
            # first: the actor's backward frees its activations before these recompute
            x_global_privileged = checkpoint(
                run_pass, x_rows_privileged, x_ordered_privileged, use_reentrant=False
            )[1]
            x_refined, x_global = run_pass(x_rows_actor, x_ordered_actor)
        else:
            # One transformer call for both passes (the rollout forward is launch-bound):
            # every state's actor row, then each privileged row again
            x_refined, x_global = run_pass(
                torch.cat([x_rows_actor, x_rows_privileged]),
                torch.cat([x_ordered_actor, x_ordered_privileged]),
            )
            batch_size = x_rows_actor.shape[0]
            x_global, x_global_privileged = x_global[:batch_size], x_global[batch_size:]
            x_refined = X_Entity(x_refined.x[:batch_size], x_refined.x_slice_kind[:batch_size])

        return X_CoreOutput(
            global_=x_global,
            global_value=x_global.index_copy(0, x_rows_privileged, x_global_privileged),
            entities=x_refined,
        )

    def _pass(
        self,
        x_game_state: X_GameState,
        x_entities: X_Entity,
        x_map_readout: torch.Tensor,
        x_pending: torch.Tensor,
        x_depth: torch.Tensor,
        x_rows: torch.Tensor,
        x_ordered: torch.Tensor,
    ) -> tuple[X_Entity, torch.Tensor]:
        """The transformer and the global context over the states `x_rows` (a state
        may repeat), adding the draw depths on the rows where `x_ordered`. Returns the
        refined window and the global vector, one row per entry of `x_rows`."""
        x_window = X_Entity(x_entities.x[x_rows], x_entities.x_slice_kind[x_rows])
        x_refined, x_global = self._entity_transformer(
            self._embed_identity(x_window, x_depth[x_rows], x_ordered)
        )
        return x_refined, self._global_context(
            x_game_state, x_refined, x_global, x_map_readout, x_pending, x_rows
        )

    def _embed_identity(
        self, x_entities: X_Entity, x_depth: torch.Tensor, x_ordered: torch.Tensor
    ) -> X_Entity:
        # Every token's identity comes from its kind: the kind's type row, plus the
        # shared CARD_REWARD identity on the reward kinds
        x_idxs = x_entities.x_slice_kind
        x_is_reward = (x_idxs >= _REWARD_LO) & (x_idxs <= _REWARD_HI)
        # Pads borrow type row 0; they are masked and zeroed downstream
        x_type = self._type_emb(x_idxs.clamp(min=0))
        # The draw tokens of an ordered row also carry their depth
        x_is_ordered = (x_idxs == SliceKind.CARD_PILE_DRAW) & x_ordered.unsqueeze(1)
        x_position = self._draw_depth_emb(x_depth) * x_is_ordered.unsqueeze(-1)
        return X_Entity(
            x_entities.x + x_type + self._reward_emb(x_is_reward.long()) + x_position, x_idxs
        )

    def _global_context(
        self,
        x_game_state: X_GameState,
        x_refined: X_Entity,
        x_global: torch.Tensor,
        x_map_readout: torch.Tensor,
        x_pending: torch.Tensor,
        x_rows: torch.Tensor,
    ) -> torch.Tensor:
        # The one character token per state, found by its kind: exactly one -> (B, D)
        x_character = x_refined.x[x_refined.x_slice_kind == SliceKind.CHARACTER]
        # The per-state inputs follow `x_rows`: a privileged row repeats its state's inputs
        return self._global_projection(
            torch.cat(
                [
                    x_global,
                    x_character,
                    x_map_readout[x_rows],
                    self._get_token_counts(x_refined.x_slice_kind),
                    x_game_state.energy[x_rows],
                    x_game_state.focus[x_rows],
                    x_game_state.ascension[x_rows],
                    x_game_state.map_meta[x_rows],
                    x_game_state.reward_meta[x_rows],
                    x_game_state.shop_meta[x_rows],
                    x_game_state.event_meta[x_rows],
                    x_pending[x_rows],
                ],
                dim=1,
            )
        )

    def _get_token_counts(self, x_idxs: torch.Tensor) -> torch.Tensor:
        """Per-kind token counts, sqrt-normalized by the kind's block width (a count
        never exceeds its block, so the ratio is in [0, 1])."""
        x_count = (x_idxs.unsqueeze(-1) == self._count_kinds).sum(dim=1).float()
        return (x_count / self._count_norm).sqrt()
