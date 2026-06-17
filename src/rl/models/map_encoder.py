import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.constants import MAP_HEIGHT
from src.rl.constants import MAP_WIDTH
from src.rl.encoding.map_ import MAP_NUM_ROOM_KINDS
from src.rl.encoding.map_ import _NUM_CHANNELS
from src.rl.encoding.map_ import _NUM_EDGE_CHANNELS

_N_NODES = MAP_HEIGHT * MAP_WIDTH
_POS_CHANNEL = _NUM_CHANNELS - 1  # current-position bit (last grid channel)


def _build_child_idx() -> torch.Tensor:
    """(_N_NODES, _NUM_EDGE_CHANNELS) long: for node n=(y,x) and relative-edge channel
    c (deltas {-1, 0, +1}), the flattened child node index in the next row, or _N_NODES
    (a padding column dropped after the scatter) when the child is off-grid. Topology only,
    so it's a fixed buffer; per-map edge existence rides on the grid's edge channels."""
    child = torch.full((_N_NODES, _NUM_EDGE_CHANNELS), _N_NODES, dtype=torch.long)
    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            n = y * MAP_WIDTH + x
            if y + 1 < MAP_HEIGHT:
                for c in range(_NUM_EDGE_CHANNELS):
                    xc = x + (c - 1)  # channels are deltas {-1, 0, +1}
                    if 0 <= xc < MAP_WIDTH:
                        child[n, c] = (y + 1) * MAP_WIDTH + xc
    return child


class MapGNN(nn.Module):
    """Graph encoder for the map DAG. Message passing aggregates each node's children
    (next-row edges + self-loop), so K = MAP_HEIGHT layers give every node full downstream
    lookahead; residual + LayerNorm per layer curb oversmoothing at that depth.

    Only forward (not-yet-visited) rooms are valid — visited/current floors are sunk and
    excluded from both message passing and the readout. The adjacency is materialized
    transiently from the grid's relative-edge channels (a dense N*N child matrix + self
    loops, row-normalized); at N=105 the BLAS bmm beats a banded gather on CPU.

    Emits (a) per-column ROOM token embeddings (the next-row selectable rooms, gathered at
    `room_node_idx`) for the entity transformer / RoomSelect pointer, and (b) a masked-mean
    graph readout (forward rooms only) for the global context.
    """

    def __init__(self, num_layers: int, hidden: int, dim_entity: int):
        super().__init__()
        self._in_proj = nn.Linear(_NUM_CHANNELS, hidden)
        self._layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(num_layers))
        self._norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(num_layers))
        self._room_proj = nn.Linear(hidden, dim_entity)

        self.register_buffer("_child_idx", _build_child_idx(), persistent=False)
        row_of = (torch.arange(_N_NODES) // MAP_WIDTH).float()
        self.register_buffer("_row_of_node", row_of, persistent=False)

    def forward(
        self, grid: torch.Tensor, room_node_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """grid (B, MAP_HEIGHT, MAP_WIDTH, _NUM_CHANNELS), room_node_idx (B, MAP_WIDTH) long.
        Returns (room_tokens (B, MAP_WIDTH, dim_entity), readout (B, hidden))."""
        b = grid.shape[0]
        nodes = grid.reshape(b, _N_NODES, _NUM_CHANNELS)

        # Envs sit on one map node for many steps, so (grid, room_idx) is highly redundant
        # across a minibatch: run the GNN on unique inputs once and gather back (exact),
        # mirroring the old CNN's dedup.
        flat = torch.cat([nodes.reshape(b, -1), room_node_idx.float()], dim=1)
        uniq, inverse = torch.unique(flat, dim=0, return_inverse=True)
        rep = torch.zeros(uniq.shape[0], dtype=torch.long, device=grid.device)
        rep[inverse] = torch.arange(b, device=grid.device)
        room_u, read_u = self._run(nodes[rep], room_node_idx[rep])
        return room_u[inverse], read_u[inverse]

    def _run(
        self, nodes: torch.Tensor, room_node_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = self._forward_valid(nodes)  # (U, N) forward rooms only
        a = self._adjacency(nodes)  # (U, N, N) row-normalized child + self

        h = self._in_proj(nodes) * valid.unsqueeze(-1)
        for lin, norm in zip(self._layers, self._norms):
            h = norm(h + F.relu(lin(torch.bmm(a, h))))
            h = h * valid.unsqueeze(-1)

        # ROOM tokens: gather the next-row nodes per column, zeroed where no legal room
        room_valid = room_node_idx >= 0  # (U, MAP_WIDTH)
        gather_idx = room_node_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, h.shape[-1])
        room_h = torch.gather(h, 1, gather_idx) * room_valid.unsqueeze(-1)
        room_tokens = self._room_proj(room_h) * room_valid.unsqueeze(-1)

        # Whole-(forward-)map readout: masked mean over valid nodes
        denom = valid.sum(1, keepdim=True).clamp(min=1).float()
        readout = (h * valid.unsqueeze(-1)).sum(1) / denom
        return room_tokens, readout

    def _adjacency(self, nodes: torch.Tensor) -> torch.Tensor:
        """Row-normalized child adjacency (+ self-loops) from the grid's edge channels.
        Off-grid edges (top row -> boss) scatter to a padding column that's dropped."""
        u = nodes.shape[0]
        edges = nodes[..., MAP_NUM_ROOM_KINDS : MAP_NUM_ROOM_KINDS + _NUM_EDGE_CHANNELS]
        a = torch.zeros(u, _N_NODES, _N_NODES + 1, device=nodes.device, dtype=nodes.dtype)
        a.scatter_(2, self._child_idx.unsqueeze(0).expand(u, -1, -1), edges)
        a = a[:, :, :_N_NODES] + torch.eye(_N_NODES, device=nodes.device, dtype=nodes.dtype)
        return a / a.sum(-1, keepdim=True).clamp(min=1.0)

    def _forward_valid(self, nodes: torch.Tensor) -> torch.Tensor:
        """Valid = a real room AND strictly ahead of the current floor (visited/current
        rows are sunk). At the act start (no position bit) every row is ahead. At the
        off-grid boss there's no position bit either, but no forward rooms exist there so
        the readout is degenerate-but-harmless."""
        room_exists = nodes[..., :MAP_NUM_ROOM_KINDS].sum(-1) > 0  # (U, N)
        pos = nodes[..., _POS_CHANNEL]  # (U, N) one-hot current node (or all-zero)
        has_pos = pos.sum(-1) > 0  # (U,)
        cur_row = (pos * self._row_of_node).sum(-1)  # (U,) current floor, 0 if none
        cur_row = torch.where(has_pos, cur_row, cur_row.new_full((), -1.0))
        forward = self._row_of_node.unsqueeze(0) > cur_row.unsqueeze(1)  # (U, N)
        return room_exists & forward
