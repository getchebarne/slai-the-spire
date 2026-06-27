import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.constants import MAP_HEIGHT
from src.rl.constants import MAP_WIDTH
from src.rl.encoding.map_ import CH_EDGES
from src.rl.encoding.map_ import CH_POSITION
from src.rl.encoding.map_ import NUM_CHANNELS
from src.rl.encoding.map_ import NUM_EDGE_CHANNELS
from src.rl.types import TPadded


_NUM_NODES = MAP_HEIGHT * MAP_WIDTH


def _build_child_idx() -> torch.Tensor:
    t_child = torch.full((_NUM_NODES, NUM_EDGE_CHANNELS), _NUM_NODES, dtype=torch.long)
    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            hw = y * MAP_WIDTH + x  # Flat index
            if y + 1 < MAP_HEIGHT:
                for c in range(NUM_EDGE_CHANNELS):
                    xc = x + (c - 1)  # {-1, 0, +1}
                    if 0 <= xc < MAP_WIDTH:
                        t_child[hw, c] = (y + 1) * MAP_WIDTH + xc
    return t_child


class MapGNN(nn.Module):
    def __init__(self, num_layers: int, hidden: int):
        super().__init__()
        self._in_proj = nn.Linear(NUM_CHANNELS, hidden)
        self._layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(num_layers))
        self._norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(num_layers))

        self.register_buffer("_child_idx", _build_child_idx(), persistent=False)
        self.register_buffer(
            "_row_of_node", (torch.arange(_NUM_NODES) // MAP_WIDTH).float(), persistent=False
        )

    def forward(
        self, t_grid: torch.Tensor, t_room_node_idx: torch.Tensor, t_room_mask: torch.Tensor
    ) -> tuple[TPadded, torch.Tensor]:
        batch_size = t_grid.shape[0]

        # Flatten grid from (B, H, W, C) -> (B, H * W, C)
        t_nodes = t_grid.reshape(batch_size, _NUM_NODES, NUM_CHANNELS)

        # The game sits on one map node for many steps, so (grid, room_idx) is highly redundant
        # across a minibatch: run the GNN on unique inputs once and gather back (exact)
        t_cat_flat = torch.cat(
            [
                t_nodes.reshape(batch_size, -1),  # (B, H * W * C)
                t_room_node_idx.float(),  # (B, W)
            ],
            dim=1,
        )  # (B, H * W * C + W)
        t_uniq, t_inverse = torch.unique(t_cat_flat, dim=0, return_inverse=True)
        # t_uniq: (U, H * W * C + W)
        # t_inverse: (B,) / t_inverse[b] ∈ [0, U). Maps unique ID to batch sample index

        t_rep = torch.zeros(t_uniq.shape[0], dtype=torch.long, device=t_grid.device)
        t_rep[t_inverse] = torch.arange(batch_size, device=t_grid.device)

        # Encode the unique maps, scatter the room features back, then mask padding columns with
        # the encoder's next-row room mask
        t_room_uniq, t_readout_uniq = self._run(t_nodes[t_rep], t_room_node_idx[t_rep])
        t_room_features = t_room_uniq[t_inverse] * t_room_mask.unsqueeze(-1)
        return TPadded(t_room_features, t_room_mask), t_readout_uniq[t_inverse]

    def _run(
        self, t_nodes: torch.Tensor, t_room_node_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_valid = self._forward_valid(t_nodes)  # (U, H * W) / Forward rooms only
        t_adj = self._adjacency(t_nodes)  # (U, H * W, H * W) / Row-normalized children

        # Message passing: aggregate children per layer, masked to forward rooms
        t_hidden = self._in_proj(t_nodes) * t_valid.unsqueeze(-1)
        for lin, norm in zip(self._layers, self._norms):
            t_hidden = norm(t_hidden + F.relu(lin(torch.bmm(t_adj, t_hidden))))
            t_hidden = t_hidden * t_valid.unsqueeze(-1)

        # Gather the next-row node features per column (the caller masks padding columns)
        t_gather_idx = (
            t_room_node_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, t_hidden.shape[-1])
        )
        t_room_h = torch.gather(t_hidden, 1, t_gather_idx)

        # Map readout: masked mean over valid nodes
        t_denom = t_valid.sum(1, keepdim=True).clamp(min=1).float()
        t_readout = (t_hidden * t_valid.unsqueeze(-1)).sum(1) / t_denom

        return t_room_h, t_readout

    def _forward_valid(self, t_nodes: torch.Tensor) -> torch.Tensor:
        """Computes the valid rooms matrix (higher than current floor and not None)"""

        t_room_exists = t_nodes.sum(-1) > 0  # (U, H * W)
        t_pos = t_nodes[:, :, CH_POSITION]  # Current-position bit: (U, H * W)
        t_row_current = (t_pos * self._row_of_node).sum(-1)  # (U,) / Current floor, 0 if none
        t_row_current = torch.where(
            condition=t_pos.sum(-1) > 0,  # (U,)
            input=t_row_current,  # (U,)
            other=t_row_current.new_full((), -1.0),  # (U,)
        )
        t_forward = self._row_of_node.unsqueeze(0) > t_row_current.unsqueeze(1)  # (U, H * W)
        return t_room_exists & t_forward  # (U, H * W)

    def _adjacency(self, t_nodes: torch.Tensor) -> torch.Tensor:
        num_unique = t_nodes.shape[0]

        t_edges = t_nodes[:, :, CH_EDGES]
        t_adj = torch.zeros(
            num_unique,
            _NUM_NODES,
            _NUM_NODES + 1,
            device=t_nodes.device,
            dtype=t_nodes.dtype,
        )
        t_adj = torch.scatter(
            t_adj,
            dim=2,
            index=self._child_idx.unsqueeze(0).expand(num_unique, -1, -1),
            src=t_edges,
        )
        # Drop the off-grid padding column, then row-normalize
        t_adj = t_adj[:, :, :_NUM_NODES]
        return t_adj / t_adj.sum(-1, keepdim=True).clamp(min=1.0)
