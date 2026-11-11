import torch
import torch.nn as nn

from src.rl.constants import MAP_TOKEN_KIND_MAX_TOKENS
from src.rl.constants import TOKEN_KINDS_CARD_REWARDS
from src.rl.encoding import ENCODING_DIM_CONTEXT
from src.rl.models.entity_projector import EntityProjector
from src.rl.models.entity_transformer import DrawOrderLayer
from src.rl.models.entity_transformer import EntityTransformer
from src.rl.models.map_encoder import MapGNN
from src.rl.models.spread_embedding import SpreadEmbedding
from src.rl.types import CoreOutput
from src.rl.types import Jagged
from src.rl.types import StateBatch
from src.rl.types import TokenKind


# Count features: one per token kind except the Character singleton (a constant 1.0), each
# normalized by its token kind's cap (a count never exceeds it), in window order
_COUNT_TOKEN_KINDS = tuple(
    token_kind for token_kind in TokenKind if token_kind is not TokenKind.CHARACTER
)


class Core(nn.Module):
    # The buffers __init__ registers, typed for mypy
    _x_count_token_kinds: torch.Tensor
    _x_count_norm: torch.Tensor
    _x_is_reward: torch.Tensor

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
            + ENCODING_DIM_CONTEXT  # Raw per-state features, in their established column order
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
            "_x_count_token_kinds", torch.tensor(_COUNT_TOKEN_KINDS), persistent=False
        )
        self.register_buffer(
            "_x_count_norm",
            torch.tensor(
                [float(MAP_TOKEN_KIND_MAX_TOKENS[token_kind]) for token_kind in _COUNT_TOKEN_KINDS]
            ).view(1, -1),
            persistent=False,
        )

        # The per-bundle token kinds carry their own type rows; this shared vector
        # is the global CARD_REWARD identity summed on top of all of them
        self._x_reward_identity = nn.Parameter(torch.empty(dim_entity))
        nn.init.normal_(self._x_reward_identity)  # N(0, 1), as the type rows

        # Per token kind, whether it is a Reward bundle's; derived, so excluded from checkpoints
        self.register_buffer(
            "_x_is_reward",
            torch.tensor([token_kind in TOKEN_KINDS_CARD_REWARDS for token_kind in TokenKind]),
            persistent=False,
        )

    def init_token_weights(self) -> None:
        """Initialize the projector's token feature ranges after the general linear init."""
        self._entity_projector.init_token_weights()

    def forward(self, state_batch: StateBatch, with_value: bool = True) -> CoreOutput:
        """Compute the core's output.

        Parameters
        ----------
        state_batch : StateBatch
            The game states.
        with_value : bool, default True
            False runs the draw-order layer only on the states the policy reads, for callers that
            need no value, and leaves `x_value_context` the policy's.

        Returns
        -------
        CoreOutput
            The policy heads' and the value head's global vectors, the refined token embeddings
            grouped by state, and their cached index.
        """

        # Map GNN: each unique Map's node features, each state's Map, and the whole-graph readout
        x_hidden_map, x_map_of_state, x_map_readout = self._map_gnn(state_batch.x_map_grid)

        # Token embeddings grouped by state, their index, the draw-order layer's states, and the
        # embedded pending halt
        projected, index, draw_order, x_effect_pending = self._entity_projector(
            state_batch, x_hidden_map, x_map_of_state
        )

        # One transformer pass over every state, without the draw depths
        refined, x_global_token = self._entity_transformer(
            Jagged(
                x_lengths=projected.x_lengths,
                rows=self._embed_identity(projected.rows, index.x_kind),
            ),
            index.x_state,
        )

        # Per-state inputs to the global context: the same for the transformer and the layer, so
        # cat'd once
        x_state_inputs = torch.cat(
            [
                x_map_readout,
                self._get_token_counts(index.x_kind_count),
                state_batch.x_context,
                x_effect_pending,
            ],
            dim=1,
        )

        # The one Character token per state, its CHARACTER row
        x_character_row = index.x_kind_row[:, TokenKind.CHARACTER]  # (B,)
        x_global = self._global_context(
            refined.rows[x_character_row], x_global_token, x_state_inputs
        )

        # The layer's states, whose pile shows an order: the Frozen Eye ones, then the rest unless
        # `with_value` is False, with their rows and pile rows
        x_states = draw_order.x_states
        num_rows, num_pile = draw_order.num_rows, draw_order.num_pile
        if not with_value:
            x_states = x_states[: draw_order.num_states_visible]
            num_rows, num_pile = draw_order.num_rows_visible, draw_order.num_pile_visible
        if x_states.shape[0] == 0:
            return CoreOutput(
                tokens=refined,
                index=index,
                x_policy_context=x_global,
                x_value_context=x_global,
            )

        # The value head reads the layer on every one of its states
        x_rows, x_refined_layer, x_global_token_layer = self._draw_order_layer(
            refined, x_global_token, index, x_states, num_rows, num_pile
        )

        # Each layer state's Character among the layer's rows: the one CHARACTER row of each of
        # its states, state after state
        x_character_layer = torch.nonzero_static(
            index.x_kind[x_rows] == TokenKind.CHARACTER, size=x_states.shape[0]
        ).squeeze(1)
        x_global_layer = self._global_context(
            x_refined_layer[x_character_layer], x_global_token_layer, x_state_inputs[x_states]
        )
        x_global_value = x_global.index_copy(0, x_states, x_global_layer)

        # The policy reads it on the Frozen Eye states: their order-aware rows written back into
        # the window
        num_states_visible = draw_order.num_states_visible
        if num_states_visible > 0:
            num_rows_visible = draw_order.num_rows_visible
            x_global = x_global.index_copy(
                0, x_states[:num_states_visible], x_global_layer[:num_states_visible]
            )
            refined = Jagged(
                x_lengths=refined.x_lengths,
                rows=refined.rows.index_copy(
                    0, x_rows[:num_rows_visible], x_refined_layer[:num_rows_visible]
                ),
            )
        return CoreOutput(
            tokens=refined,
            index=index,
            x_policy_context=x_global,
            x_value_context=x_global_value,
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

    def _embed_identity(
        self, x_projected: torch.Tensor, x_token_kind: torch.Tensor
    ) -> torch.Tensor:
        # Every token's identity comes from its token kind: the token kind's type row, plus the
        # shared CARD_REWARD identity on the Reward token kinds
        x_is_reward = self._x_is_reward[x_token_kind]
        x_type = SpreadEmbedding.apply(self._type_emb.weight, x_token_kind)
        return x_projected + x_type + x_is_reward.unsqueeze(1) * self._x_reward_identity

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
        x_count = x_token_kind_count[:, self._x_count_token_kinds].float()
        return (x_count / self._x_count_norm).sqrt()
