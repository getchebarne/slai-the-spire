from dataclasses import replace

import torch
import torch.nn as nn

from src.rl.batches import get_entity_kind_rows
from src.rl.batches import get_row_owner
from src.rl.constants import NUM_ACTION_TYPES
from src.rl.constants import MAX_TOKENS_OF_TOKEN_KIND
from src.rl.constants import TOKEN_KINDS_CARD_REWARDS
from src.rl.constants import TOKEN_KINDS_IN_WINDOW_ORDER
from src.rl.encoding import ENCODING_DIM_CONTEXT
from src.rl.models.entity_projector import EntityProjector
from src.rl.models.entity_transformer import EntityTransformer
from src.rl.models.map_encoder import MapGNN
from src.rl.models.spread_embedding import SpreadEmbedding
from src.rl.types import CoreOutput
from src.rl.types import Dense
from src.rl.types import EntityKind
from src.rl.types import Live
from src.rl.types import States
from src.rl.types import TokenKind


# Count features: one per token kind except the Character singleton (a constant 1.0), each
# normalized by its token kind's cap (a count never exceeds it), in window order
_COUNT_TOKEN_KINDS = tuple(
    token_kind
    for token_kind in TOKEN_KINDS_IN_WINDOW_ORDER
    if token_kind is not TokenKind.CHARACTER
)


class Core(nn.Module):
    _x_count_token_kind: torch.Tensor
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
            + NUM_ACTION_TYPES         # The legal action types after the masks, 0/1
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
            "_x_count_token_kind", torch.tensor(_COUNT_TOKEN_KINDS), persistent=False
        )
        self.register_buffer(
            "_x_count_norm",
            torch.tensor(
                [float(MAX_TOKENS_OF_TOKEN_KIND[token_kind]) for token_kind in _COUNT_TOKEN_KINDS]
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

    def forward(self, states: States[Live]) -> CoreOutput:
        """Compute the core's output.

        Parameters
        ----------
        states : States[Live]
            The game states.

        Returns
        -------
        CoreOutput
            The refined tokens grouped by state with their legal masks, and each state's global
            vector, legal action types and counts.
        """

        tokens = states.tokens
        num_tokens = tokens.x_token_kind.shape[0]
        batch_size = tokens.x_count.shape[0]

        # Each token's state
        x_token_state = get_row_owner(tokens.x_count, num_tokens)  # (N,)

        # Map GNN: the Room tokens' node features and the whole-graph readout
        num_rooms = int(states.x_total_cpu[:, TokenKind.ROOMS].sum())
        x_room_hidden, x_map_readout = self._map_gnn(states.x_map_grid, num_rooms)

        # Token embeddings grouped by state, and the embedded pending halt
        tokens_projected, x_effect_pending = self._entity_projector(states, x_room_hidden)

        # One transformer pass over every state
        tokens_refined, x_global_token = self._entity_transformer(
            replace(
                tokens_projected,
                features=Dense(
                    x_value=self._embed_identity(
                        tokens_projected.features.x_value, tokens.x_token_kind.long()
                    )
                ),
            ),
            x_token_state,
        )

        # Each state's global vector
        x_character_row = get_entity_kind_rows(
            tokens.x_entity_kind, EntityKind.CHARACTER, batch_size
        )  # (B,)
        x_global = self._global_projection(
            torch.cat(
                [
                    x_global_token,
                    tokens_refined.features.x_value[x_character_row],
                    x_map_readout,
                    self._get_token_counts(states.x_total),
                    states.x_context,
                    x_effect_pending,
                    states.x_legal_action_type.to(states.x_context.dtype),
                ],
                dim=1,
            )
        )

        # The input tokens' Entity kinds and legal masks stay with the refined tokens
        return CoreOutput(
            tokens=tokens_refined,
            x_token_state=x_token_state,
            x_global=x_global,
            x_legal_action_type=states.x_legal_action_type,
            x_total=states.x_total,
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

    def _get_token_counts(self, x_total: torch.Tensor) -> torch.Tensor:
        """Compute the per-token-kind token counts, sqrt-normalized by the token kind's cap.

        A count never exceeds its cap, so the ratio is in [0, 1].

        Parameters
        ----------
        x_total : torch.Tensor
            (B, C) int32. Each state's totals; TokenKind indexes its token-kind columns.

        Returns
        -------
        torch.Tensor
            (B, len(_COUNT_TOKEN_KINDS)) float32. Each state's normalized counts, one per token
            kind except the Character.
        """
        x_count = x_total[:, self._x_count_token_kind].float()

        return (x_count / self._x_count_norm).sqrt()
