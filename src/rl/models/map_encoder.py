import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.constants import MAP_HEIGHT
from src.rl.constants import MAP_WIDTH
from src.rl.encoding.map_ import MAP_NUM_ROOM_KINDS
from src.rl.encoding.map_ import NUM_CHANNELS
from src.rl.encoding.map_ import NUM_EDGE_CHANNELS
from src.rl.types import TPadded

_NUM_NODES = MAP_HEIGHT * MAP_WIDTH


def _build_child_idx() -> torch.Tensor:
    """(_NUM_NODES, NUM_EDGE_CHANNELS) long: for node n=(y,x) and relative-edge channel
    c (deltas {-1, 0, +1}), the flattened child node index in the next row, or _NUM_NODES
    (a padding column dropped after the scatter) when the child is off-grid. Topology only,
    so it's a fixed buffer; per-map edge existence rides on the grid's edge channels."""
    t_child = torch.full((_NUM_NODES, NUM_EDGE_CHANNELS), _NUM_NODES, dtype=torch.long)
    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            n = y * MAP_WIDTH + x
            if y + 1 < MAP_HEIGHT:
                for c in range(NUM_EDGE_CHANNELS):
                    xc = x + (c - 1)  # channels are deltas {-1, 0, +1}
                    if 0 <= xc < MAP_WIDTH:
                        t_child[n, c] = (y + 1) * MAP_WIDTH + xc
    return t_child


class MapGNN(nn.Module):
    """Graph encoder for the map DAG. Message passing aggregates each node's children
    (next-row edges + self-loop), so K = MAP_HEIGHT layers give every node full downstream
    lookahead; residual + LayerNorm per layer curb oversmoothing at that depth.

    Only forward (not-yet-visited) rooms are valid — visited/current floors are sunk and
    excluded from both message passing and the readout. The adjacency is materialized
    transiently from the grid's relative-edge channels (a dense N*N child matrix + self
    loops, row-normalized); at N=105 the BLAS bmm beats a banded gather on CPU.

    Emits (a) per-column next-row room node features (the selectable rooms, gathered at
    `room_node_idx`; EntityProjector lifts them to the entity-token width), and (b) a masked-mean
    graph readout (forward rooms only) for the global context.
    """

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
        self, t_grid: torch.Tensor, t_room_node_idx: torch.Tensor
    ) -> tuple[TPadded, torch.Tensor]:
        """t_grid (B, MAP_HEIGHT, MAP_WIDTH, NUM_CHANNELS), t_room_node_idx (B, MAP_WIDTH) long.
        Returns the next-row room node features as a TPadded (x (B, MAP_WIDTH, hidden), mask True
        where the column has a selectable room) + the forward-map readout (B, hidden)."""
        batch_size = t_grid.shape[0]
        t_nodes = t_grid.reshape(batch_size, _NUM_NODES, NUM_CHANNELS)

        # Envs sit on one map node for many steps, so (grid, room_idx) is highly redundant
        # across a minibatch: run the GNN on unique inputs once and gather back (exact),
        # mirroring the old CNN's dedup.
        t_flat = torch.cat([t_nodes.reshape(batch_size, -1), t_room_node_idx.float()], dim=1)
        t_uniq, t_inverse = torch.unique(t_flat, dim=0, return_inverse=True)
        t_rep = torch.zeros(t_uniq.shape[0], dtype=torch.long, device=t_grid.device)
        t_rep[t_inverse] = torch.arange(batch_size, device=t_grid.device)

        # Encode the unique maps, then scatter the results back over the original batch
        t_room_u, t_valid_u, t_readout_u = self._run(t_nodes[t_rep], t_room_node_idx[t_rep])
        return TPadded(t_room_u[t_inverse], t_valid_u[t_inverse]), t_readout_u[t_inverse]

    def _run(
        self, t_nodes: torch.Tensor, t_room_node_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_valid = self._forward_valid(t_nodes)  # (U, N) forward rooms only
        t_a = self._adjacency(t_nodes)  # (U, N, N) row-normalized child + self

        # Message passing: aggregate children + self per layer, masked to forward rooms
        t_h = self._in_proj(t_nodes) * t_valid.unsqueeze(-1)
        for lin, norm in zip(self._layers, self._norms):
            t_h = norm(t_h + F.relu(lin(torch.bmm(t_a, t_h))))
            t_h = t_h * t_valid.unsqueeze(-1)

        # Gather the next-row node features per column, zeroed where no legal room (EntityProjector
        # lifts them to dim_entity)
        t_room_valid = t_room_node_idx >= 0  # (U, MAP_WIDTH)
        t_gather_idx = t_room_node_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, t_h.shape[-1])
        t_room_h = torch.gather(t_h, 1, t_gather_idx) * t_room_valid.unsqueeze(-1)

        # Whole-(forward-)map readout: masked mean over valid nodes
        t_denom = t_valid.sum(1, keepdim=True).clamp(min=1).float()
        t_readout = (t_h * t_valid.unsqueeze(-1)).sum(1) / t_denom
        return t_room_h, t_room_valid, t_readout

    def _adjacency(self, t_nodes: torch.Tensor) -> torch.Tensor:
        """Row-normalized child adjacency (+ self-loops) from the grid's edge channels.
        Off-grid edges (top row -> boss) scatter to a padding column that's dropped."""
        num_unique = t_nodes.shape[0]
        t_edges = t_nodes[:, :, MAP_NUM_ROOM_KINDS : MAP_NUM_ROOM_KINDS + NUM_EDGE_CHANNELS]
        t_a = torch.zeros(
            num_unique, _NUM_NODES, _NUM_NODES + 1, device=t_nodes.device, dtype=t_nodes.dtype
        )
        t_a.scatter_(2, self._child_idx.unsqueeze(0).expand(num_unique, -1, -1), t_edges)

        # Drop the off-grid padding column, add self-loops, then row-normalize
        t_a = t_a[:, :, :_NUM_NODES] + torch.eye(
            _NUM_NODES, device=t_nodes.device, dtype=t_nodes.dtype
        )
        return t_a / t_a.sum(-1, keepdim=True).clamp(min=1.0)

    def _forward_valid(self, t_nodes: torch.Tensor) -> torch.Tensor:
        """Valid = a real room AND strictly ahead of the current floor (visited/current
        rows are sunk). At the act start (no position bit) every row is ahead. At the
        off-grid boss there's no position bit either, but no forward rooms exist there so
        the readout is degenerate-but-harmless."""
        t_room_exists = t_nodes[:, :, :MAP_NUM_ROOM_KINDS].sum(-1) > 0  # (U, N)
        t_pos = t_nodes[:, :, NUM_CHANNELS - 1]  # current-position bit: (U, N) one-hot node, or 0
        t_has_pos = t_pos.sum(-1) > 0  # (U,)
        t_cur_row = (t_pos * self._row_of_node).sum(-1)  # (U,) current floor, 0 if none
        t_cur_row = torch.where(t_has_pos, t_cur_row, t_cur_row.new_full((), -1.0))
        t_forward = self._row_of_node.unsqueeze(0) > t_cur_row.unsqueeze(1)  # (U, N)
        return t_room_exists & t_forward
