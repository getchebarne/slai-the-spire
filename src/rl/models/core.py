import torch
import torch.nn as nn

from src.rl.constants import MAP_TOKEN_KIND_MAX_TOKENS
from src.rl.constants import TOKEN_KINDS_CARD_REWARDS
from src.rl.encoding import ENCODING_DIM_CONTEXT
from src.rl.models.entity_projector import EntityProjector
from src.rl.models.entity_transformer import EntityTransformer
from src.rl.models.map_encoder import MapGNN
from src.rl.models.spread_embedding import SpreadEmbedding
from src.rl.types import CoreOutput
from src.rl.types import Jagged
from src.rl.types import StateBatch
from src.rl.types import StateCounts
from src.rl.types import TokenKind
from src.rl.types import TokensRefined


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

        # Token identity: one row per `TokenKind`, looked up by the window's token kind row
        self._token_kind_emb = nn.Embedding(len(TokenKind), dim_entity)

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

        # The per-bundle token kinds carry their own type rows
        self._x_reward_identity = nn.Parameter(torch.empty(dim_entity))
        nn.init.normal_(self._x_reward_identity)  # N(0, 1), as the type rows

        # Per token kind, whether it is a Reward bundle
        self.register_buffer(
            "_x_is_reward",
            torch.tensor([token_kind in TOKEN_KINDS_CARD_REWARDS for token_kind in TokenKind]),
            persistent=False,
        )

    def init_token_weights(self) -> None:
        """Initialize the projector's token feature ranges after the general linear init."""
        self._entity_projector.init_token_weights()

    def forward(self, state_batch: StateBatch, state_counts_cpu: StateCounts) -> CoreOutput:
        """Compute the core's output.

        Parameters
        ----------
        state_batch : StateBatch
            The game states.
        state_counts_cpu : StateCounts
            The states' counts on the CPU, `state_batch.state_counts`.

        Returns
        -------
        CoreOutput
            The refined tokens grouped by state with their legal masks, and each state's global
            vector, legal action types and counts.
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
        jag_projected, x_effect_pending = self._entity_projector(state_batch, x_room_hidden)

        # One transformer pass over every state
        jag_refined, x_global_token = self._entity_transformer(
            Jagged(
                x_lengths=jag_projected.x_lengths,
                rows=self._embed_identity(
                    jag_projected.rows, state_batch.tokens.rows.x_token_kind.long()
                ),
            ),
            x_state,
        )

        # Each state's global vector
        x_is_character = tokens.rows.x_token_kind == TokenKind.CHARACTER  # (N,)
        x_character_row = torch.nonzero_static(x_is_character, size=batch_size).squeeze(1)  # (B,)
        x_global = self._global_projection(
            torch.cat(
                [
                    x_global_token,
                    jag_refined.rows[x_character_row],
                    x_map_readout,
                    self._get_token_counts(state_batch.state_counts.x_token_kind_count),
                    state_batch.x_context,
                    x_effect_pending,
                ],
                dim=1,
            )
        )

        # The input tokens' Entity kinds and legal masks stay with the refined tokens
        tokens_refined = TokensRefined(
            x_embedding=jag_refined.rows,
            x_entity_kind=tokens.rows.x_entity_kind,
            x_legal_selection=tokens.rows.x_legal_selection,
            x_legal_target=tokens.rows.x_legal_target,
            x_needs_target=tokens.rows.x_needs_target,
        )

        return CoreOutput(
            tokens=Jagged(x_lengths=jag_refined.x_lengths, rows=tokens_refined),
            x_global=x_global,
            x_legal_action_type=state_batch.x_legal_action_type,
            state_counts=state_batch.state_counts,
        )

    def _embed_identity(
        self, x_projected: torch.Tensor, x_token_kind: torch.Tensor
    ) -> torch.Tensor:
        """Add each token's identity to its embedding.

        The identity is the token kind's type row, plus the shared Reward identity when the token
        kind is a Reward bundle.

        Parameters
        ----------
        x_projected : torch.Tensor
            (N, dim_entity). The projector's token embeddings.
        x_token_kind : torch.Tensor
            (N,) long. Each token's kind.

        Returns
        -------
        torch.Tensor
            (N, dim_entity). The embeddings with their identities added.
        """
        x_token_kind_emb = SpreadEmbedding.apply(self._token_kind_emb.weight, x_token_kind)

        return x_projected + x_token_kind_emb + self._x_is_reward[x_token_kind].unsqueeze(1) * self._x_reward_identity

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
