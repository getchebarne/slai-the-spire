from dataclasses import replace

import torch
import torch.nn as nn

from src.rl.batches import get_entity_kind_rows
from src.rl.batches import get_row_owner
from src.rl.constants import COLLECTION_KINDS_CARD_REWARDS
from src.rl.constants import COLLECTION_KINDS_IN_WINDOW_ORDER
from src.rl.constants import NUM_ACTION_TYPES
from src.rl.constants import MAX_TOKENS_OF_COLLECTION_KIND
from src.rl.encoding import ENCODING_DIM_CONTEXT
from src.rl.models.entity_projector import EntityProjector
from src.rl.models.entity_transformer import EntityTransformer
from src.rl.models.map_encoder import MapGNN
from src.rl.models.spread_embedding import SpreadEmbedding
from src.rl.types import CollectionKind
from src.rl.types import CoreOutput
from src.rl.types import Dense
from src.rl.types import EntityKind
from src.rl.types import Live
from src.rl.types import MapCache
from src.rl.types import MapDedup
from src.rl.types import States


# Count features: one per collection kind except the Character singleton (a constant 1.0), each
# normalized by its collection kind's cap (a count never exceeds it), in window order
_COUNT_COLLECTION_KINDS = tuple(
    collection_kind
    for collection_kind in COLLECTION_KINDS_IN_WINDOW_ORDER
    if collection_kind is not CollectionKind.CHARACTER
)


class Core(nn.Module):
    _x_count_collection_kind: torch.Tensor
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
        dim_map: int,
    ) -> None:
        super().__init__()

        # Entity projector
        self._entity_projector = EntityProjector(dim_entity, dim_map)

        # Entity transformer
        self._entity_transformer = EntityTransformer(
            dim_entity, transformer_dim_ff, transformer_num_heads, transformer_num_blocks
        )

        # Token identity: one row per `CollectionKind`, summed over the token's collection kinds
        self._collection_kind_emb = nn.Embedding(len(CollectionKind), dim_entity)

        # Map GNN
        self._map_gnn = MapGNN(gnn_num_layers, dim_map)

        # Global context: global token + Character + Map readout + collection-kind counts
        # + raw flats
        dim_global_input = (
            dim_entity                 # Global token (attention-aggregated Entities)
            + dim_entity               # Character (refined singleton)
            + dim_map                  # Map GNN readout (whole-graph summary)
            + len(_COUNT_COLLECTION_KINDS)  # Per-collection-kind token counts, sqrt-normalized
                                            # by their caps
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
            "_x_count_collection_kind", torch.tensor(_COUNT_COLLECTION_KINDS), persistent=False
        )
        self.register_buffer(
            "_x_count_norm",
            torch.tensor(
                [
                    float(MAX_TOKENS_OF_COLLECTION_KIND[collection_kind])
                    for collection_kind in _COUNT_COLLECTION_KINDS
                ]
            ).view(1, -1),
            persistent=False,
        )

        # The per-bundle collection kinds carry their own type rows
        self._x_reward_identity = nn.Parameter(torch.empty(dim_entity))
        nn.init.normal_(self._x_reward_identity)  # N(0, 1), as the type rows

        # Per collection kind, whether it is a Reward bundle
        self.register_buffer(
            "_x_is_reward",
            torch.tensor(
                [
                    collection_kind in COLLECTION_KINDS_CARD_REWARDS
                    for collection_kind in CollectionKind
                ]
            ),
            persistent=False,
        )

    def init_token_weights(self) -> None:
        """Initialize the projector's token feature ranges after the general linear init."""
        self._entity_projector.init_token_weights()

    def forward(
        self,
        states: States[Live],
        map_cache: MapCache | None = None,
        map_dedup: MapDedup | None = None,
    ) -> CoreOutput:
        """Compute the core's output.

        Parameters
        ----------
        states : States[Live]
            The game states.
        map_cache : MapCache | None, default None
            The Map GNN's outputs to reuse and update within a rollout (see `MapGNN.forward`).
        map_dedup : MapDedup | None, default None
            The Maps the Map GNN runs, planned on the host for an update (see `MapGNN.forward`).

        Returns
        -------
        CoreOutput
            The refined tokens grouped by state with their legal masks, and each state's global
            vector, legal action types and counts.
        """

        tokens = states.tokens
        num_tokens = tokens.x_collection_kind.shape[0]
        batch_size = tokens.x_count.shape[0]

        # Each token's state
        x_token_state = get_row_owner(tokens.x_count, num_tokens)  # (N,)

        # Map GNN: the Room tokens' cell features and the whole-graph readout
        num_rooms = int(states.x_total_cpu[:, CollectionKind.ROOMS].sum())
        x_room_hidden, x_map_readout = self._map_gnn(
            states.x_map_grid, num_rooms, map_cache, map_dedup
        )

        # Token embeddings grouped by state, and the embedded pending halt
        tokens_projected, x_effect_pending = self._entity_projector(states, x_room_hidden)

        # The tokens' memberships, their collection kinds, counted on the host
        num_memberships = int(states.x_total_cpu[:, : len(CollectionKind)].sum())

        # One transformer pass over every state
        tokens_refined, x_global_token = self._entity_transformer(
            replace(
                tokens_projected,
                features=Dense(
                    x_value=self._embed_identity(
                        tokens_projected.features.x_value,
                        tokens.x_collection_kind,
                        num_memberships,
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
        self, x_projected: torch.Tensor, x_collection_kind: torch.Tensor, num_memberships: int
    ) -> torch.Tensor:
        """Add each token's identity to its embedding.

        The identity sums the type rows of the collection kinds holding the token, plus the shared
        Reward identity once for each of them that is a Reward bundle.

        Parameters
        ----------
        x_projected : torch.Tensor
            (N, dim_entity). The projector's token embeddings.
        x_collection_kind : torch.Tensor
            (N, len(CollectionKind)) uint8. Each token's collection kinds, 1 at each collection
            holding it.
        num_memberships : int
            The tokens' total number of collection kinds, their memberships, as the host knows it:
            no size is read back from the device.

        Returns
        -------
        torch.Tensor
            (N, dim_entity). The embeddings with their identities added.
        """
        num_tokens, num_collection_kinds = x_collection_kind.shape

        # Each token's lowest collection kind's type row, for all tokens
        x_collection_kind_lowest = x_collection_kind.argmax(1)  # (N,)
        x_collection_kind_emb = SpreadEmbedding.apply(
            self._collection_kind_emb.weight, x_collection_kind_lowest
        )

        # The other memberships, of tokens held by several collections, counted on the host
        num_memberships_other = num_memberships - num_tokens
        if num_memberships_other > 0:
            # Each other membership's token and collection kind: the multi-hot minus the lowest
            x_collection_kind_other = x_collection_kind.scatter(
                1, x_collection_kind_lowest.unsqueeze(1), 0
            )
            x_membership = torch.nonzero_static(
                x_collection_kind_other.flatten(), size=num_memberships_other
            )
            x_membership = x_membership.squeeze(1)  # (num_memberships_other,)

            # Token and column of each collection membership bit
            x_membership_token = x_membership // num_collection_kinds
            x_membership_collection_kind = x_membership % num_collection_kinds

            # Their type rows, added into their tokens' rows in place
            x_membership_emb = SpreadEmbedding.apply(
                self._collection_kind_emb.weight, x_membership_collection_kind
            )  # (num_memberships_other, dim_entity)
            x_collection_kind_emb.index_add_(0, x_membership_token, x_membership_emb)

        # Each token's Reward bundles among its collection kinds
        x_count_reward = (
            (x_collection_kind * self._x_is_reward).sum(1).to(x_collection_kind_emb.dtype)
        )

        return (
            x_projected
            + x_collection_kind_emb
            + x_count_reward.unsqueeze(1) * self._x_reward_identity
        )

    def _get_token_counts(self, x_total: torch.Tensor) -> torch.Tensor:
        """Compute the per-collection-kind token counts, sqrt-normalized by their caps.

        A count never exceeds its cap, so the ratio is in [0, 1].

        Parameters
        ----------
        x_total : torch.Tensor
            (B, C) int32. Each state's totals; CollectionKind indexes its collection-kind columns.

        Returns
        -------
        torch.Tensor
            (B, len(_COUNT_COLLECTION_KINDS)) float32. Each state's normalized counts, one per
            collection kind except the Character.
        """
        x_count = x_total[:, self._x_count_collection_kind].float()

        return (x_count / self._x_count_norm).sqrt()
