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
from src.rl.types import StateCounts
from src.rl.types import TokenKind


# Count features: one per token kind except the Character singleton (a constant 1.0), each
# normalized by its token kind's cap (a count never exceeds it), in window order
_COUNT_TOKEN_KINDS = tuple(
    token_kind for token_kind in TokenKind if token_kind is not TokenKind.CHARACTER
)


class Core(nn.Module):
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

        # Entity projector
        self._entity_projector = EntityProjector(dim_entity, map_encoder_dim)

        # Entity transformer
        self._entity_transformer = EntityTransformer(
            dim_entity, transformer_dim_ff, transformer_num_heads, transformer_num_blocks
        )

        # Draw order layer
        self._draw_order_layer = DrawOrderLayer(
            dim_entity, transformer_dim_ff, transformer_num_heads
        )

        # Token identity: one row per `TokenKind`, looked up by the window's token kind row
        self._type_emb = nn.Embedding(len(TokenKind), dim_entity)

        # Map GNN
        self._map_gnn = MapGNN(gnn_num_layers, map_encoder_dim)

        # Global context: global token + Character + Map readout + token-kind counts + raw flats
        dim_global_input = (
            dim_entity                 # Global token (attention-aggregated Entities)
            + dim_entity               # Character (refined singleton)
            + map_encoder_dim          # Map GNN readout (whole-graph summary)
            + len(_COUNT_TOKEN_KINDS)  # Per-token-kind token counts, sqrt-normalized by their caps
            + ENCODING_DIM_CONTEXT     # Raw per-state features, in their established column order
            + dim_entity               # Embedded pending-halt Effect (EntityProjector)
        )

        # Global output
        self._global_projection = nn.Sequential(
            nn.Linear(dim_global_input, dim_global),
            nn.ReLU(),
            nn.Linear(dim_global, dim_global),
            nn.LayerNorm(dim_global),
        )

        # Count features
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

    def forward(
        self, state_batch: StateBatch, state_counts_cpu: StateCounts, with_value: bool = True
    ) -> CoreOutput:
        """Compute the core's output.

        Parameters
        ----------
        state_batch : StateBatch
            The game states.
        state_counts_cpu : StateCounts
            The states' counts on the CPU, `state_batch.state_counts`.
        with_value : bool, default True
            False runs the draw-order layer only on the states the policy reads, for callers that
            need no value, and leaves `x_value_context` the policy's.

        Returns
        -------
        CoreOutput
            The policy heads' and the value head's global vectors, and the refined token
            embeddings grouped by state.
        """

        tokens = state_batch.tokens
        num_tokens = tokens.rows.x_token_kind.shape[0]
        batch_size = tokens.x_lengths.shape[0]

        # Each token's state
        x_state = torch.repeat_interleave(tokens.x_lengths, output_size=num_tokens)  # (N,)

        # Map GNN: the Room tokens' node features and the whole-graph readout
        num_rooms = int(state_counts_cpu.x_token_kind_count[:, TokenKind.ROOMS].sum())
        x_room_hidden, x_map_readout = self._map_gnn(state_batch.x_map_grid, num_rooms)

        # Token embeddings grouped by state, and the embedded pending halt
        projected, x_effect_pending = self._entity_projector(state_batch, x_room_hidden)

        # One transformer pass over every state, without the draw depths
        refined, x_global_token = self._entity_transformer(
            Jagged(
                x_lengths=projected.x_lengths,
                rows=self._embed_identity(
                    projected.rows, state_batch.tokens.rows.x_token_kind.long()
                ),
            ),
            x_state,
        )

        # Per-state inputs to the global context: the same for the transformer and the layer, so
        # cat'd once
        x_state_inputs = torch.cat(
            [
                x_map_readout,
                self._get_token_counts(state_batch.state_counts.x_token_kind_count),
                state_batch.x_context,
                x_effect_pending,
            ],
            dim=1,
        )

        # The one Character token per state, its CHARACTER row
        x_is_character = tokens.rows.x_token_kind == TokenKind.CHARACTER  # (N,)
        x_character_row = torch.nonzero_static(x_is_character, size=batch_size).squeeze(1)  # (B,)
        x_global = self._global_context(
            refined.rows[x_character_row], x_global_token, x_state_inputs
        )

        # The draw-order layer, on the states whose pile shows an order
        refined, x_policy_context, x_value_context = self._apply_draw_order_layer(
            state_batch,
            refined,
            x_global_token,
            x_global,
            x_state_inputs,
            state_counts_cpu,
            with_value,
        )
        return CoreOutput(
            tokens=refined,
            x_policy_context=x_policy_context,
            x_value_context=x_value_context,
        )

    def _apply_draw_order_layer(
        self,
        state_batch: StateBatch,
        refined: Jagged[torch.Tensor],
        x_global_token: torch.Tensor,
        x_global: torch.Tensor,
        x_state_inputs: torch.Tensor,
        state_counts_cpu: StateCounts,
        with_value: bool,
    ) -> tuple[Jagged[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Run the draw-order layer and write its output where the value and the policy read it.

        The value reads the layer on every state whose pile shows an order, the policy only under
        Frozen Eye.

        Parameters
        ----------
        state_batch : StateBatch
            The game states.
        refined : Jagged[torch.Tensor]
            The transformer's refined rows, one segment per state.
        x_global_token : torch.Tensor
            (B, dim_entity). The transformer's global tokens.
        x_global : torch.Tensor
            (B, dim_global). Each state's global vector, without the draw order.
        x_state_inputs : torch.Tensor
            (B, ...). Each state's per-state inputs to the global context.
        state_counts_cpu : StateCounts
            The states' counts on the CPU, `state_batch.state_counts`.
        with_value : bool
            False runs the layer only on the states the policy reads.

        Returns
        -------
        refined : Jagged[torch.Tensor]
            The refined rows, order-aware on the Frozen Eye states.
        x_policy_context : torch.Tensor
            (B, dim_global). The policy heads' global vectors.
        x_value_context : torch.Tensor
            (B, dim_global). The value head's global vectors; the policy's when `with_value` is
            False.
        """
        # The states whose pile shows an order, as the device and as the host hold them
        x_visible, x_hidden = _ordered_piles(state_batch.state_counts)
        x_visible_cpu, x_hidden_cpu = _ordered_piles(state_counts_cpu)

        # The layer's states: the Frozen Eye ones, then the rest unless `with_value` is False
        num_states_policy = int(x_visible_cpu.sum())
        x_states = torch.nonzero_static(x_visible, size=num_states_policy).squeeze(1)
        x_layer_cpu = x_visible_cpu
        if with_value:
            num_states_hidden = int(x_hidden_cpu.sum())
            x_states_hidden = torch.nonzero_static(x_hidden, size=num_states_hidden).squeeze(1)
            x_states = torch.cat([x_states, x_states_hidden])
            x_layer_cpu = x_visible_cpu | x_hidden_cpu
        if x_states.shape[0] == 0:
            return refined, x_global, x_global

        # Those states' tokens, of every token kind, and their draw tokens, summed on the host
        x_token_kind_count_cpu = state_counts_cpu.x_token_kind_count
        num_rows = int(x_token_kind_count_cpu[x_layer_cpu].sum())
        num_pile = int(x_token_kind_count_cpu[x_layer_cpu, TokenKind.CARD_PILE_DRAW].sum())

        # The value head reads the layer on every one of its states
        x_rows, x_refined_layer, x_global_token_layer = self._draw_order_layer(
            refined,
            x_global_token,
            state_batch.state_counts.x_token_kind_count,
            x_states,
            num_rows,
            num_pile,
        )

        # Each layer state's Character among the layer's rows: the one CHARACTER row of each of
        # its states, state after state
        x_character_layer = torch.nonzero_static(
            state_batch.tokens.rows.x_token_kind[x_rows] == TokenKind.CHARACTER,
            size=x_states.shape[0],
        ).squeeze(1)
        x_global_layer = self._global_context(
            x_refined_layer[x_character_layer], x_global_token_layer, x_state_inputs[x_states]
        )
        x_value_context = x_global.index_copy(0, x_states, x_global_layer)

        # The policy reads it on the Frozen Eye states: their order-aware rows written back into
        # the window
        x_policy_context = x_global
        if num_states_policy > 0:
            num_rows_policy = int(x_token_kind_count_cpu[x_visible_cpu].sum())
            x_policy_context = x_global.index_copy(
                0, x_states[:num_states_policy], x_global_layer[:num_states_policy]
            )
            refined = Jagged(
                x_lengths=refined.x_lengths,
                rows=refined.rows.index_copy(
                    0, x_rows[:num_rows_policy], x_refined_layer[:num_rows_policy]
                ),
            )
        return refined, x_policy_context, x_value_context

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
            (B, len(TokenKind)) int16. Each state's tokens of each token kind.

        Returns
        -------
        torch.Tensor
            (B, len(_COUNT_TOKEN_KINDS)) float32. Each state's normalized counts, one per token
            kind except the Character.
        """
        x_count = x_token_kind_count[:, self._x_count_token_kinds].float()
        return (x_count / self._x_count_norm).sqrt()


def _ordered_piles(state_counts: StateCounts) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the states whose draw pile shows an order: 2 or more draw tokens.

    One definition for both homes of the counts: the device's copy gives the states, the host's
    their sizes.

    Parameters
    ----------
    state_counts : StateCounts
        The states' counts, on the device or on the CPU.

    Returns
    -------
    x_visible : torch.Tensor
        (B,) bool. The ordered piles under Frozen Eye, which the policy reads too.
    x_hidden : torch.Tensor
        (B,) bool. The other ordered piles, which only the value reads.
    """
    x_ordered = state_counts.x_token_kind_count[:, TokenKind.CARD_PILE_DRAW] >= 2
    x_draw_order_visible = state_counts.x_draw_order_visible
    return x_ordered & x_draw_order_visible, x_ordered & ~x_draw_order_visible
