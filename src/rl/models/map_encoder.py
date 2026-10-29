import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.constants import MAP_HEIGHT
from src.rl.constants import MAP_WIDTH
from src.rl.encoding.map_ import EDGE_DELTAS
from src.rl.encoding.map_ import NUM_CHANNELS
from src.rl.encoding.map_ import NUM_EDGE_CHANNELS
from src.rl.encoding.map_ import OFF_MAP_POSITION
from src.rl.encoding.map_ import SLICE_MAP_EDGES
from src.rl.encoding.map_ import SLICE_MAP_ROOM_KIND


_NUM_NODES = MAP_HEIGHT * MAP_WIDTH

# Dedup pays only on batches that repeat Maps (update minibatches, not rollout batches), and
# is exact either way; 512 sits between the largest rollout batch and the smallest minibatch
_DEDUP_MIN_BATCH = 512

# The dedup's packed rows: the grid's 0/1 cells 63 to an int64 word
_GRID_BITS = _NUM_NODES * NUM_CHANNELS


def _build_child_idx() -> torch.Tensor:
    x_child = torch.full((_NUM_NODES, NUM_EDGE_CHANNELS), _NUM_NODES, dtype=torch.long)
    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            hw = y * MAP_WIDTH + x  # Flat index
            if y + 1 < MAP_HEIGHT:
                for c, delta in enumerate(EDGE_DELTAS):
                    xc = x + delta  # {-1, 0, +1}
                    if 0 <= xc < MAP_WIDTH:
                        x_child[hw, c] = (y + 1) * MAP_WIDTH + xc
    return x_child


def _pack_rows(x_nodes: torch.Tensor) -> torch.Tensor:
    """Pack each grid into int64 words that sort as the float grid does.

    This way unique() groups and orders them alike at a fraction of the cost: the 0/1 cells most
    significant first, every word non-negative.

    Parameters
    ----------
    x_nodes : torch.Tensor
        (B, H * W, C) float32. Each state's Map grid, flattened to nodes.

    Returns
    -------
    torch.Tensor
        (B, words) long. Each grid's 0/1 cells, 63 to a word.
    """
    batch_size, device = x_nodes.shape[0], x_nodes.device
    x_bit_shifts = torch.arange(62, -1, -1, device=device)
    x_bits = F.pad(x_nodes.reshape(batch_size, -1).long(), (0, -_GRID_BITS % 63))
    return (x_bits.view(batch_size, -1, 63) << x_bit_shifts).sum(-1)  # (B, words)


class MapGNN(nn.Module):
    """Message passing over the act's Map.

    Each layer, a Room reads the mean of its children in the next row, so `num_layers` layers see
    that many rows ahead. Only the Rooms above the position row take part, or every Room when
    there is no position bit (before the act's first Room, and in the boss and boss treasure
    Rooms).

    Returns the node features (U, H * W, hidden) of each Map it ran on, the unique Maps on a
    batch of _DEDUP_MIN_BATCH states or more and every state's on a smaller one, and each state's
    Map (B,); then the readout (B, hidden): the mean over the Rooms that take part, zero when
    none does.

    Parameters
    ----------
    num_layers : int
        The number of message-passing layers.
    hidden : int
        The width of the node features and the readout.
    """

    def __init__(self, num_layers: int, hidden: int) -> None:
        super().__init__()
        self._in_proj = nn.Linear(NUM_CHANNELS, hidden)
        self._layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(num_layers))
        self._norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(num_layers))

        self.register_buffer("_child_idx", _build_child_idx(), persistent=False)
        self.register_buffer(
            "_row_of_node", (torch.arange(_NUM_NODES) // MAP_WIDTH).float(), persistent=False
        )

        # The GNN's computation as a plain function, which actor_critic.compile_hot_paths swaps for
        # its compiled version on the learner's model
        self.message_passing = message_passing

    def forward(self, x_grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the message passing over `x_grid`'s Maps.

        Parameters
        ----------
        x_grid : torch.Tensor
            (B, H, W, C) float32. Each state's Map.

        Returns
        -------
        x_hidden_map : torch.Tensor
            (U, H * W, hidden). The node features of each Map the GNN ran on: the unique Maps
            on a batch of _DEDUP_MIN_BATCH states or more, every state's on a smaller one.
        x_map_of_state : torch.Tensor
            (B,) long. Each state's Map, its index in `x_hidden_map`.
        x_map_readout : torch.Tensor
            (B, hidden). Each state's readout.
        """
        batch_size = x_grid.shape[0]

        # Flatten grid from (B, H, W, C) -> (B, H * W, C)
        x_nodes = x_grid.reshape(batch_size, _NUM_NODES, NUM_CHANNELS)

        if batch_size < _DEDUP_MIN_BATCH:
            x_hidden, x_readout = self.message_passing(self, x_nodes)
            return x_hidden, torch.arange(batch_size, device=x_grid.device), x_readout

        # The game sits on one Map node for many steps, so its grid repeats across a minibatch:
        # run the GNN on unique grids once and gather back (exact)
        x_packed = _pack_rows(x_nodes)  # (B, words)

        # x_uniq: (U, words)
        # x_inverse: (B,) / x_inverse[b] ∈ [0, U). Maps each batch sample to its unique ID
        x_uniq, x_inverse = torch.unique(x_packed, dim=0, return_inverse=True)

        x_rep = torch.zeros(x_uniq.shape[0], dtype=torch.long, device=x_grid.device)
        x_rep[x_inverse] = torch.arange(batch_size, device=x_grid.device)

        # Encode the unique Maps, and gather the readout back
        x_hidden, x_readout_uniq = self.message_passing(self, x_nodes[x_rep])
        return x_hidden, x_inverse, x_readout_uniq[x_inverse]

    def _forward_valid(self, x_nodes: torch.Tensor) -> torch.Tensor:
        """Compute the valid Rooms matrix (higher than current floor and not None).

        Parameters
        ----------
        x_nodes : torch.Tensor
            (U, H * W, C) float32. Each Map's nodes.

        Returns
        -------
        torch.Tensor
            (U, H * W) bool. Whether each node is a valid Room.
        """

        # The encoder sets one Room-kind bit per Room, none elsewhere, and at most one
        # position bit, on the current Room
        x_room_exists = x_nodes[:, :, SLICE_MAP_ROOM_KIND].sum(-1) > 0  # (U, H * W)
        x_position = x_nodes[:, :, OFF_MAP_POSITION]  # Current-position bit: (U, H * W)
        x_row_current = (x_position * self._row_of_node).sum(-1)  # (U,) / Current floor, 0 if none
        x_row_current = torch.where(
            condition=x_position.sum(-1) > 0,  # (U,)
            input=x_row_current,  # (U,)
            other=torch.full(
                (), -1.0, dtype=x_row_current.dtype, device=x_row_current.device
            ),  # (U,)
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


def message_passing(gnn: MapGNN, x_nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run `gnn` over the Maps.

    Parameters
    ----------
    gnn : MapGNN
        The GNN to run.
    x_nodes : torch.Tensor
        (U, H * W, C) float32. The Maps.

    Returns
    -------
    x_hidden : torch.Tensor
        (U, H * W, hidden). Each node's features.
    x_readout : torch.Tensor
        (U, hidden). The readout: each Map's mean over its valid nodes, zero when it has none.
    """
    x_valid = gnn._forward_valid(x_nodes)  # (U, H * W) / Forward Rooms only
    x_adj = gnn._adjacency(x_nodes)  # (U, H * W, H * W) / Row-normalized children

    # Message passing: aggregate children per layer, masked to forward Rooms. The dense bmm
    # beats a sparse gather-sum here: at HW=105 it is launch-bound, not FLOP-bound
    x_hidden = gnn._in_proj(x_nodes) * x_valid.unsqueeze(-1)
    for lin, norm in zip(gnn._layers, gnn._norms):
        x_hidden = norm(x_hidden + F.relu(lin(torch.bmm(x_adj, x_hidden))))
        x_hidden = x_hidden * x_valid.unsqueeze(-1)

    # Map readout: masked mean over valid nodes
    x_denom = x_valid.sum(1, keepdim=True).clamp(min=1).float()
    x_readout = x_hidden.sum(1) / x_denom  # Already zero off x_valid: masked above

    return x_hidden, x_readout
