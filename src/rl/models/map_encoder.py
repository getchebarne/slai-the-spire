import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.constants import MAP_HEIGHT
from src.rl.constants import MAP_WIDTH
from src.rl.encoding.map_ import NUM_CHANNELS
from src.rl.encoding.map_ import NUM_EDGE_CHANNELS
from src.rl.encoding.map_ import OFF_MAP_POSITION
from src.rl.encoding.map_ import SLICE_MAP_EDGES
from src.rl.types import X_Entity


_NUM_NODES = MAP_HEIGHT * MAP_WIDTH

# Dedup pays only when the batch actually repeats maps (update minibatches, 2048 rows);
# rollout batches (64-256) are ~all-distinct, where unique() is pure overhead — measured
# −0.2 s/iter on the RTX 3090 (2026-07 probe). Exact either way; 512 sits between the
# largest rollout batch and the smallest update minibatch.
_DEDUP_MIN_BATCH = 512


def _build_child_idx() -> torch.Tensor:
    x_child = torch.full((_NUM_NODES, NUM_EDGE_CHANNELS), _NUM_NODES, dtype=torch.long)
    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            hw = y * MAP_WIDTH + x  # Flat index
            if y + 1 < MAP_HEIGHT:
                for c in range(NUM_EDGE_CHANNELS):
                    xc = x + (c - 1)  # {-1, 0, +1}
                    if 0 <= xc < MAP_WIDTH:
                        x_child[hw, c] = (y + 1) * MAP_WIDTH + xc
    return x_child


class MapGNN(nn.Module):
    def __init__(self, num_layers: int, hidden: int) -> None:
        super().__init__()
        self._in_proj = nn.Linear(NUM_CHANNELS, hidden)
        self._layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(num_layers))
        self._norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(num_layers))

        self.register_buffer("_child_idx", _build_child_idx(), persistent=False)
        self.register_buffer(
            "_row_of_node", (torch.arange(_NUM_NODES) // MAP_WIDTH).float(), persistent=False
        )

    def forward(
        self, x_grid: torch.Tensor, x_room_node_idx: torch.Tensor, x_room_idxs: torch.Tensor
    ) -> tuple[X_Entity, torch.Tensor]:
        batch_size = x_grid.shape[0]
        x_room_mask = x_room_idxs >= 0

        # Flatten grid from (B, H, W, C) -> (B, H * W, C)
        x_nodes = x_grid.reshape(batch_size, _NUM_NODES, NUM_CHANNELS)

        if batch_size < _DEDUP_MIN_BATCH:
            x_room_h, x_readout = self._run(x_nodes, x_room_node_idx)
            return X_Entity(x_room_h * x_room_mask.unsqueeze(-1), x_room_idxs), x_readout

        # The game sits on one map node for many steps, so (grid, room_idx) is highly redundant
        # across a minibatch: run the GNN on unique inputs once and gather back (exact)
        x_cat_flat = torch.cat(
            [
                x_nodes.reshape(batch_size, -1),  # (B, H * W * C)
                x_room_node_idx.float(),  # (B, W)
            ],
            dim=1,
        )  # (B, H * W * C + W)
        x_uniq, x_inverse = torch.unique(x_cat_flat, dim=0, return_inverse=True)
        # x_uniq: (U, H * W * C + W)
        # x_inverse: (B,) / x_inverse[b] ∈ [0, U). Maps unique ID to batch sample index

        x_rep = torch.zeros(x_uniq.shape[0], dtype=torch.long, device=x_grid.device)
        x_rep[x_inverse] = torch.arange(batch_size, device=x_grid.device)

        # Encode the unique maps, scatter the room features back, then mask padding columns with
        # the encoder's next-row room mask
        x_room_uniq, x_readout_uniq = self._run(x_nodes[x_rep], x_room_node_idx[x_rep])
        x_room_features = x_room_uniq[x_inverse] * x_room_mask.unsqueeze(-1)
        return X_Entity(x_room_features, x_room_idxs), x_readout_uniq[x_inverse]

    def _run(
        self, x_nodes: torch.Tensor, x_room_node_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_valid = self._forward_valid(x_nodes)  # (U, H * W) / Forward rooms only
        x_adj = self._adjacency(x_nodes)  # (U, H * W, H * W) / Row-normalized children

        # Message passing: aggregate children per layer, masked to forward rooms.
        # The dense bmm looks wasteful (≤3 children/room) but a sparse gather-sum measured
        # SLOWER on GPU (8.4 vs 7.5 s/iter, 2026-07): at HW=105 the bmm is launch-bound,
        # not FLOP-bound, and the gather path adds kernels + memory traffic per layer.
        x_hidden = self._in_proj(x_nodes) * x_valid.unsqueeze(-1)
        for lin, norm in zip(self._layers, self._norms):
            x_hidden = norm(x_hidden + F.relu(lin(torch.bmm(x_adj, x_hidden))))
            x_hidden = x_hidden * x_valid.unsqueeze(-1)

        # Gather the next-row node features per column (the caller masks padding columns)
        x_gather_idx = (
            x_room_node_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, x_hidden.shape[-1])
        )
        x_room_h = torch.gather(x_hidden, 1, x_gather_idx)

        # Map readout: masked mean over valid nodes
        x_denom = x_valid.sum(1, keepdim=True).clamp(min=1).float()
        x_readout = (x_hidden * x_valid.unsqueeze(-1)).sum(1) / x_denom

        return x_room_h, x_readout

    def _forward_valid(self, x_nodes: torch.Tensor) -> torch.Tensor:
        """Computes the valid rooms matrix (higher than current floor and not None)"""

        x_room_exists = x_nodes.sum(-1) > 0  # (U, H * W)
        x_pos = x_nodes[:, :, OFF_MAP_POSITION]  # Current-position bit: (U, H * W)
        x_row_current = (x_pos * self._row_of_node).sum(-1)  # (U,) / Current floor, 0 if none
        x_row_current = torch.where(
            condition=x_pos.sum(-1) > 0,  # (U,)
            input=x_row_current,  # (U,)
            other=x_row_current.new_full((), -1.0),  # (U,)
        )
        x_forward = self._row_of_node.unsqueeze(0) > x_row_current.unsqueeze(1)  # (U, H * W)
        return x_room_exists & x_forward  # (U, H * W)

    def _adjacency(self, x_nodes: torch.Tensor) -> torch.Tensor:
        num_unique = x_nodes.shape[0]

        x_edges = x_nodes[:, :, SLICE_MAP_EDGES]
        x_adj = torch.zeros(
            num_unique,
            _NUM_NODES,
            _NUM_NODES + 1,
            device=x_nodes.device,
            dtype=x_nodes.dtype,
        )
        x_adj = torch.scatter(
            x_adj,
            dim=2,
            index=self._child_idx.unsqueeze(0).expand(num_unique, -1, -1),
            src=x_edges,
        )
        # Drop the off-grid padding column, then row-normalize
        x_adj = x_adj[:, :, :_NUM_NODES]
        return x_adj / x_adj.sum(-1, keepdim=True).clamp(min=1.0)
