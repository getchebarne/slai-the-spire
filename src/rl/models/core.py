import torch
import torch.nn as nn

from src.rl.encoding.energy import ENCODING_DIM_ENERGY
from src.rl.encoding.event import ENCODING_DIM_EVENT_META
from src.rl.encoding.focus import ENCODING_DIM_FOCUS
from src.rl.encoding.map_ import ENCODING_DIM_MAP_META
from src.rl.encoding.reward import ENCODING_DIM_REWARD_META
from src.rl.encoding.shop import ENCODING_DIM_SHOP_META
from src.rl.encoding.state import ENCODING_DIM_ASCENSION
from src.rl.models.entity_projector import EntityProjector
from src.rl.models.entity_transformer import DrawOrderLayer
from src.rl.models.entity_transformer import EntityTransformer
from src.rl.models.map_encoder import MapGNN
from src.rl.models.spread_embedding import SpreadEmbedding
from src.rl.types import MAP_TOKEN_KIND_CAP
from src.rl.types import TOKEN_KINDS_CARD_REWARDS
from src.rl.types import TokenKind
from src.rl.types import X_CoreOutput
from src.rl.types import X_GameState


# Count features: one per token kind except the Character singleton (a constant 1.0), each
# normalized by its token kind's cap (a count never exceeds it), in window order
_COUNT_TOKEN_KINDS = tuple(
    token_kind for token_kind in TokenKind if token_kind is not TokenKind.CHARACTER
)


class Core(nn.Module):
    # The buffers __init__ registers, typed for mypy
    _count_token_kinds: torch.Tensor
    _count_norm: torch.Tensor
    _is_reward: torch.Tensor

    def __init__(
        self,
        dim_entity: int,
        dim_global: int,
        transformer_dim_ff: int,
        transformer_num_heads: int,
        transformer_num_blocks: int,
        gnn_num_layers: int,
        map_encoder_dim: int,
    ) -> None:
        super().__init__()

        self._entity_projector = EntityProjector(dim_entity, map_encoder_dim)
        self._entity_transformer = EntityTransformer(
            dim_entity, transformer_dim_ff, transformer_num_heads, transformer_num_blocks
        )

        # Draw order: the value head reads the layer on every ordered pile, the policy only under
        # Frozen Eye
        self._draw_order_layer = DrawOrderLayer(
            dim_entity, transformer_dim_ff, transformer_num_heads
        )

        # Token identity: one row per TokenKind, looked up by the window's token kind row
        self._type_emb = nn.Embedding(len(TokenKind), dim_entity)
        self._map_gnn = MapGNN(gnn_num_layers, map_encoder_dim)

        # Global context: global token + Character + Map readout + token-kind counts + raw flats
        global_input_dim = (
            dim_entity  # Global token (attention-aggregated Entities)
            + dim_entity  # Character (refined singleton)
            + map_encoder_dim  # Map GNN readout (whole-graph summary)
            + len(_COUNT_TOKEN_KINDS)  # Per-token-kind token counts, sqrt-normalized by their caps
            + ENCODING_DIM_ENERGY  # Energy (raw)
            + ENCODING_DIM_FOCUS  # Focus row
            + ENCODING_DIM_ASCENSION  # Ascension thermometer (curriculum level)
            + ENCODING_DIM_MAP_META  # Floor, act, boss, Room kinds, "?"-room odds and Event pools
            + ENCODING_DIM_REWARD_META
            + ENCODING_DIM_SHOP_META  # Shop purge price triple + purged flag
            + ENCODING_DIM_EVENT_META
            + dim_entity  # Embedded pending-halt Effect (EntityProjector)
        )

        # Output LN: HeadActionType/HeadValue read this vector without their
        # own norm, so scale drift here would move logits directly
        self._global_projection = nn.Sequential(
            nn.Linear(global_input_dim, dim_global),
            nn.ReLU(),
            nn.Linear(dim_global, dim_global),
            nn.LayerNorm(dim_global),
        )

        # Count features: token kinds to count and each token kind's cap; derived, so excluded
        # from checkpoints
        self.register_buffer(
            "_count_token_kinds", torch.tensor(_COUNT_TOKEN_KINDS), persistent=False
        )
        self.register_buffer(
            "_count_norm",
            torch.tensor(
                [float(MAP_TOKEN_KIND_CAP[token_kind]) for token_kind in _COUNT_TOKEN_KINDS]
            ).view(1, -1),
            persistent=False,
        )

        # The per-bundle token kinds carry their own type rows; this shared vector
        # is the global CARD_REWARD identity summed on top of all of them
        self._reward_identity = nn.Parameter(torch.empty(dim_entity))
        nn.init.normal_(self._reward_identity)  # N(0, 1), as the type rows

        # Per token kind, whether it is a Reward bundle's; derived, so excluded from checkpoints
        self.register_buffer(
            "_is_reward",
            torch.tensor([token_kind in TOKEN_KINDS_CARD_REWARDS for token_kind in TokenKind]),
            persistent=False,
        )

    def forward(self, x_gs: X_GameState, with_value: bool = True) -> X_CoreOutput:
        """Compute the core's output.

        Parameters
        ----------
        x_gs : X_GameState
            The game states.
        with_value : bool, default True
            False runs the draw-order layer only on the states the policy reads, for callers that
            need no value, and leaves `global_value` the policy's.

        Returns
        -------
        X_CoreOutput
            The policy heads' and the value head's global vectors, the refined rows and the
            packed window's X_Window.
        """

        # Map GNN: each unique Map's node features, each state's Map, and the whole-graph readout
        x_hidden_map, x_map_of_state, x_map_readout = self._map_gnn(x_gs.map_grid)

        # The window packed (its tokens and their X_Window), the draw-order layer's states, and
        # the embedded pending halt
        x_tokens, x_window, x_draw_order, x_pending = self._entity_projector(
            x_gs, x_hidden_map, x_map_of_state
        )

        # One transformer pass over every state, without the draw depths
        x_refined, x_global_token = self._entity_transformer(
            self._embed_identity(x_tokens, x_window.x_token_kind), x_window
        )

        # Per-state inputs to the global context: the same for the transformer and the layer, so
        # cat'd once
        x_state_inputs = torch.cat(
            [
                x_map_readout,
                self._get_token_counts(x_window.x_token_kind_count),
                x_gs.energy,
                x_gs.focus,
                x_gs.ascension,
                x_gs.map_meta,
                x_gs.reward_meta,
                x_gs.shop_meta,
                x_gs.event_meta,
                x_pending,
            ],
            dim=1,
        )

        # The one Character token per state, its CHARACTER row
        x_character_row = x_window.x_token_kind_row[:, TokenKind.CHARACTER]  # (B,)
        x_global = self._global_context(x_refined[x_character_row], x_global_token, x_state_inputs)

        # The layer's states, whose pile shows an order: the Frozen Eye ones, then the rest unless
        # `with_value` is False, with their rows and pile rows
        x_states = x_draw_order.x_states
        num_rows, num_pile = x_draw_order.num_rows, x_draw_order.num_pile
        if not with_value:
            x_states = x_states[: x_draw_order.num_states_visible]
            num_rows, num_pile = x_draw_order.num_rows_visible, x_draw_order.num_pile_visible
        if x_states.shape[0] == 0:
            return X_CoreOutput(
                global_=x_global,
                global_value=x_global,
                x_tokens=x_refined,
                window=x_window,
            )

        # The value head reads the layer on every one of its states
        x_rows, x_refined_layer, x_global_token_layer = self._draw_order_layer(
            x_refined, x_global_token, x_window, x_states, num_rows, num_pile
        )

        # Each layer state's Character among the layer's rows: the one CHARACTER row of each of
        # its states, state after state
        x_character_layer = torch.nonzero_static(
            x_window.x_token_kind[x_rows] == TokenKind.CHARACTER, size=x_states.shape[0]
        ).squeeze(1)
        x_global_layer = self._global_context(
            x_refined_layer[x_character_layer], x_global_token_layer, x_state_inputs[x_states]
        )
        x_global_value = x_global.index_copy(0, x_states, x_global_layer)

        # The policy reads it on the Frozen Eye states: their order-aware rows written back into
        # the window
        num_states_visible = x_draw_order.num_states_visible
        if num_states_visible > 0:
            num_rows_visible = x_draw_order.num_rows_visible
            x_global = x_global.index_copy(
                0, x_states[:num_states_visible], x_global_layer[:num_states_visible]
            )
            x_refined = x_refined.index_copy(
                0, x_rows[:num_rows_visible], x_refined_layer[:num_rows_visible]
            )
        return X_CoreOutput(
            global_=x_global,
            global_value=x_global_value,
            x_tokens=x_refined,
            window=x_window,
        )

    def _global_context(
        self, x_character: torch.Tensor, x_global_token: torch.Tensor, x_state_inputs: torch.Tensor
    ) -> torch.Tensor:
        """Compute the global vector of each state.

        Parameters
        ----------
        x_character : torch.Tensor
            (B, dim_entity). Each state's refined Character token.
        x_global_token : torch.Tensor
            (B, dim_entity). Each state's global token.
        x_state_inputs : torch.Tensor
            (B, ...). Each state's per-state inputs to the global context.

        Returns
        -------
        torch.Tensor
            (B, dim_global). Each state's global vector.
        """
        return self._global_projection(
            torch.cat([x_global_token, x_character, x_state_inputs], dim=1)
        )

    def _embed_identity(self, x_tokens: torch.Tensor, x_token_kind: torch.Tensor) -> torch.Tensor:
        # Every token's identity comes from its token kind: the token kind's type row, plus the
        # shared CARD_REWARD identity on the Reward token kinds
        x_is_reward = self._is_reward[x_token_kind]
        x_type = SpreadEmbedding.apply(self._type_emb.weight, x_token_kind)
        return x_tokens + x_type + x_is_reward.unsqueeze(1) * self._reward_identity

    def _get_token_counts(self, x_token_kind_count: torch.Tensor) -> torch.Tensor:
        """Compute the per-token-kind token counts, sqrt-normalized by the token kind's cap.

        A count never exceeds its cap, so the ratio is in [0, 1].

        Parameters
        ----------
        x_token_kind_count : torch.Tensor
            (B, len(TokenKind)) long. Each state's tokens of each token kind.

        Returns
        -------
        torch.Tensor
            (B, len(_COUNT_TOKEN_KINDS)) float32. Each state's normalized counts, one per token
            kind except the Character.
        """
        x_count = x_token_kind_count[:, self._count_token_kinds].float()
        return (x_count / self._count_norm).sqrt()
