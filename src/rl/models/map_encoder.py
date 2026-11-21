from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.batches import dedup_rows
from src.rl.constants import MAP_HEIGHT
from src.rl.constants import MAP_WIDTH
from src.rl.constants import MIN_BATCH_DEDUP
from src.rl.encoding import EDGE_DELTAS
from src.rl.encoding import NUM_FEATURE_CHANNELS
from src.rl.encoding import OFF_MAP_POSITION
from src.rl.encoding import SLICE_MAP_EDGES
from src.rl.encoding import SLICE_MAP_ROOM_KIND
from src.rl.maps import UniqueMaps
from src.rl.types import MapDedup
from src.rl.types import MapPlan


_NUM_CELLS = MAP_HEIGHT * MAP_WIDTH


@dataclass
class RoomGraph:
    """Forward Rooms in message-passing order, grouped by their number of children."""

    x_node_cell: torch.Tensor                  # (N,) Each node's cell across the Maps.
    x_children_by_count: tuple[torch.Tensor, ...]  # (Nk, k) for k = 3, 2, 1.
    x_count: torch.Tensor                      # (U, 1) Forward Rooms in each Map.
    x_feature_row: torch.Tensor | None         # (N,) Each node's row in the projection table.


class MapGNN(nn.Module):
    """Message passing over the act's Map.

    Each layer, a Room reads the mean of its children in the next row, so `num_layers` layers see
    that many rows ahead. Only the forward Rooms take part: the Rooms above the position row, or
    every Room when there is no position bit (before the act's first Room, and in the boss and
    boss treasure Rooms).

    Takes the Entity Projector's embeddings and the graph of forward Rooms. Returns a grid of
    contextual Room embeddings and the mean embedding of each Map's forward Rooms.

    Parameters
    ----------
    num_layers : int
        The number of message-passing layers.
    dim_map : int
        The width of the cell features and the readout.
    dim_entity : int
        The width of an Entity embedding, both before the GNN and at the transformer.
    """

    _x_edge_delta: torch.Tensor
    _x_cell_features: torch.Tensor
    _x_feature_id: torch.Tensor

    def __init__(self, num_layers: int, dim_map: int, dim_entity: int) -> None:
        super().__init__()
        self._in_proj = nn.Linear(dim_entity, dim_map)
        self._layers = nn.ModuleList(nn.Linear(dim_map, dim_map) for _ in range(num_layers))
        self._norms = nn.ModuleList(nn.LayerNorm(dim_map) for _ in range(num_layers))

        self._out_proj = nn.Identity() if dim_map == dim_entity else nn.Linear(dim_map, dim_entity)

        # Every local Room feature row is one Room kind followed by binary flags. A large set
        # of Rooms projects the small table of possible rows once and looks up its embeddings.
        num_kinds = SLICE_MAP_ROOM_KIND.stop
        num_flags = NUM_FEATURE_CHANNELS - num_kinds
        num_flag_sets = 1 << num_flags
        x_id = torch.arange(num_kinds * num_flag_sets)
        x_bits = 1 << torch.arange(num_flags)
        x_features = torch.cat(
            [F.one_hot(x_id // num_flag_sets, num_kinds), (x_id[:, None] & x_bits).ne(0)], dim=1
        ).float()
        self.register_buffer("_x_cell_features", x_features, persistent=False)
        self.register_buffer(
            "_x_feature_id", torch.cat([torch.arange(num_kinds) * num_flag_sets, x_bits]).float(),
            persistent=False,
        )
        self.register_buffer("_x_edge_delta", torch.tensor(EDGE_DELTAS), persistent=False)

        # The GNN's computation as a function; `compile_hot_paths` swaps in its compiled version
        self.message_passing = message_passing

    def prepare_rooms(
        self, x_cell: torch.Tensor, np_count_room_forward: np.ndarray | None
    ) -> tuple[torch.Tensor, RoomGraph]:
        """Return the local feature rows to project and the graph that uses them.

        `x_cell` holds the selected Maps' grids, (U, H * W, C). When a Learner minibatch
        supplies the four child-count totals, finding forward Rooms needs no GPU readback.
        """
        num_maps = x_cell.shape[0]
        num_cells = num_maps * _NUM_CELLS
        x_cell = x_cell.float()
        x_is_room_forward = _is_room_forward(x_cell)
        x_count = x_is_room_forward.sum(1, keepdim=True)
        x_is_room_forward = x_is_room_forward.flatten()
        x_cell = x_cell.flatten(0, 1)
        x_count_child = _count_children(x_cell)

        if np_count_room_forward is None:
            np_count_room_forward = (
                _count_rooms_forward(x_is_room_forward, x_count_child).cpu().numpy()
            )
        num_0, num_1, num_2, num_3 = np_count_room_forward.tolist()
        num_rooms_forward = num_0 + num_1 + num_2 + num_3

        # Each forward Room's cell, most children first
        x_room_forward_cell = torch.nonzero_static(
            x_is_room_forward, size=num_rooms_forward
        ).squeeze(1)  # (N,)
        x_room_forward_cell = x_room_forward_cell[
            torch.argsort(x_count_child[x_room_forward_cell], descending=True)
        ]
        x_room_forward = x_cell[x_room_forward_cell]  # (N, C)

        # (U * H * W,) Each forward Room's index at its cell, to turn a child's cell into its index
        x_idx_room_forward = torch.zeros(num_cells, dtype=torch.long, device=x_cell.device)
        x_idx_room_forward[x_room_forward_cell] = torch.arange(
            num_rooms_forward, device=x_cell.device
        )

        # Each edge's child, by index: an edge leads to the next row, `delta` columns over
        x_edge_parent, x_idx_edge = torch.nonzero_static(
            x_room_forward[:, SLICE_MAP_EDGES], size=num_1 + 2 * num_2 + 3 * num_3
        ).unbind(1)
        x_child_cell = (
            x_room_forward_cell[x_edge_parent] + MAP_WIDTH + self._x_edge_delta[x_idx_edge]
        )
        x_child = x_idx_room_forward[x_child_cell]  # (num_edges,)

        # (Nk, k) The children of the Rooms with 3, 2 and 1 child, which come first in that order
        x_3, x_2, x_1 = x_child.split([3 * num_3, 2 * num_2, num_1])
        x_children_by_count = (x_3.view(num_3, 3), x_2.view(num_2, 2), x_1.view(num_1, 1))

        # On large graphs, identical local features share a row in the projection table.
        x_features = x_room_forward
        x_feature_row = None
        if num_rooms_forward > self._x_cell_features.shape[0]:
            x_features = self._x_cell_features
            x_feature_row = (x_room_forward * self._x_feature_id).sum(1).long()

        return x_features, RoomGraph(
            x_node_cell=x_room_forward_cell,
            x_children_by_count=x_children_by_count,
            x_count=x_count,
            x_feature_row=x_feature_row,
        )

    def forward(
        self, x_projected: torch.Tensor, graph: RoomGraph
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pass messages between projected Rooms, then return the grid and mean for each Map."""
        num_maps = graph.x_count.shape[0]
        dim_map = self._in_proj.out_features
        if graph.x_node_cell.numel() == 0:
            weight = self._in_proj.weight
            return (
                weight.new_zeros((num_maps, _NUM_CELLS, dim_map)),
                weight.new_zeros((num_maps, dim_map)),
            )

        x_local = self._in_proj(x_projected)
        if graph.x_feature_row is not None:
            x_local = x_local.index_select(0, graph.x_feature_row)
        x_nodes = self.message_passing(self, x_local, graph.x_children_by_count)

        x_hidden = x_nodes.new_zeros((num_maps * _NUM_CELLS, dim_map))
        x_hidden[graph.x_node_cell] = x_nodes
        x_hidden = x_hidden.view(num_maps, _NUM_CELLS, dim_map)
        x_readout = x_hidden.sum(1) / graph.x_count.clamp(min=1).float()

        return x_hidden, x_readout

    def select_rooms(
        self, x_hidden: torch.Tensor, x_room_map: torch.Tensor, x_room_cell: torch.Tensor
    ) -> torch.Tensor:
        """Read the contextual embedding of each selectable Room at the transformer's width."""
        return self._out_proj(x_hidden[x_room_map, x_room_cell])


def message_passing(
    gnn: MapGNN, x_hidden: torch.Tensor, x_children_by_count: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    """Run `gnn` over the forward Rooms, each Room reading the mean of its children.

    Parameters
    ----------
    gnn : MapGNN
        The GNN to run.
    x_hidden : torch.Tensor
        (N, dim_map). The forward Rooms' local embeddings, most children first.
    x_children_by_count : tuple[torch.Tensor, ...]
        (Nk, k) long, one per child count k, most children first. The next Nk Rooms' children, by
        index in `x_hidden`. The Rooms past them have no children.

    Returns
    -------
    torch.Tensor
        (N, dim_map). Each Room's features.
    """
    # The childless Rooms, which come last, read a zero mean
    num_rooms_childless = x_hidden.shape[0] - sum(
        x_child.shape[0] for x_child in x_children_by_count
    )
    x_mean_childless = torch.zeros(
        num_rooms_childless, x_hidden.shape[1], dtype=x_hidden.dtype, device=x_hidden.device
    )
    for linear, norm in zip(gnn._layers, gnn._norms):
        # (N, dim_map); mean = zero for childless Rooms
        x_mean = torch.cat(
            [x_hidden[x_child].mean(1) for x_child in x_children_by_count] + [x_mean_childless]
        )

        # Linear projection -> ReLU -> Normalize
        x_hidden = norm(x_hidden + F.relu(linear(x_mean)))

    return x_hidden


def plan_maps(maps: UniqueMaps) -> MapPlan:
    """Count each distinct Map's forward Rooms and read the plan to the host."""
    x_cell = maps.x_cell.float()  # (U, H * W, NUM_FEATURE_CHANNELS)
    x_count_room_forward = _count_rooms_forward(_is_room_forward(x_cell), _count_children(x_cell))

    # The states' Maps and the counts in one host read
    num_states = maps.x_state_map.shape[0]
    np_buffer = torch.cat([maps.x_state_map, x_count_room_forward.flatten()]).cpu().numpy()

    return MapPlan(
        np_state_map=np_buffer[:num_states],
        np_count_room_forward=np_buffer[num_states:].reshape(x_count_room_forward.shape),
    )


def dedup_maps(map_plan: MapPlan, np_idx_state: np.ndarray, device: torch.device) -> MapDedup:
    """Take one forward's `MapDedup` from its update's plan, on the host.

    A forward of MIN_BATCH_DEDUP states or more runs its unique Maps, in the plan's order, each
    from its first state; a smaller one runs every state's.

    Parameters
    ----------
    map_plan : MapPlan
        The update's plan.
    np_idx_state : np.ndarray
        (B,) int64. The forward's states, by their index among the plan's.
    device : torch.device
        The forward's device.

    Returns
    -------
    MapDedup
        The Maps the forward runs, each state's among them, and their forward Rooms with 0 to 3
        children.
    """
    batch_size = len(np_idx_state)
    np_state_map_plan = map_plan.np_state_map[np_idx_state]  # (B,)
    if batch_size < MIN_BATCH_DEDUP:
        x_rep = torch.arange(batch_size, device=device)  # Every state's own Map

        return MapDedup(
            x_rep=x_rep,
            x_state_map=x_rep,
            np_count_room_forward=map_plan.np_count_room_forward[np_state_map_plan].sum(0),
        )

    x_rep, x_state_map, np_count_room_forward = dedup_rows(
        np_state_map_plan, map_plan.np_count_room_forward, device
    )

    return MapDedup(
        x_rep=x_rep, x_state_map=x_state_map, np_count_room_forward=np_count_room_forward
    )


def _is_room_forward(x_cell: torch.Tensor) -> torch.Tensor:
    """Find the forward Rooms: the cells holding a Room above the current floor.

    Parameters
    ----------
    x_cell : torch.Tensor
        (U, H * W, C) float32. Each Map's cells.

    Returns
    -------
    torch.Tensor
        (U, H * W) bool. Whether each cell holds a forward Room.
    """
    # Each cell's row on the grid
    x_row_of_cell = torch.arange(_NUM_CELLS, device=x_cell.device) // MAP_WIDTH  # (H * W,)

    # The encoder sets one Room-kind bit per Room, none elsewhere, and at most one
    # position bit, on the current Room
    x_room_exists = x_cell[:, :, SLICE_MAP_ROOM_KIND].sum(-1) > 0  # (U, H * W)
    x_position = x_cell[:, :, OFF_MAP_POSITION]  # Current-position bit: (U, H * W)
    x_row_current = (x_position * x_row_of_cell).sum(-1)  # (U,) / Current floor, 0 if none
    x_row_current = torch.where(
        condition=x_position.sum(-1) > 0,  # (U,)
        input=x_row_current,  # (U,)
        other=torch.full(
            (), -1.0, dtype=x_row_current.dtype, device=x_row_current.device
        ),  # (U,)
    )
    x_forward = x_row_of_cell.unsqueeze(0) > x_row_current.unsqueeze(1)  # (U, H * W)

    return x_room_exists & x_forward  # (U, H * W)


def _count_children(x_cell: torch.Tensor) -> torch.Tensor:
    """Count each cell's children: its edges to the next row.

    Parameters
    ----------
    x_cell : torch.Tensor
        (..., C) float32. Map cells, along any leading axes.

    Returns
    -------
    torch.Tensor
        (...) long. Each cell's number of children.
    """
    return x_cell[..., SLICE_MAP_EDGES].sum(-1).long()


def _count_rooms_forward(
    x_is_room_forward: torch.Tensor, x_count_child: torch.Tensor
) -> torch.Tensor:
    """Count the forward Rooms with 0 to 3 children along the last axis, the cells'.

    Parameters
    ----------
    x_is_room_forward : torch.Tensor
        (..., N) bool. Whether each of N cells holds a forward Room.
    x_count_child : torch.Tensor
        (..., N) long. Each cell's number of children.

    Returns
    -------
    torch.Tensor
        (..., 4) long. The forward Rooms with 0 to 3 children.
    """
    x_count = torch.zeros(
        *x_count_child.shape[:-1],
        len(EDGE_DELTAS) + 1,
        dtype=torch.long,
        device=x_count_child.device,
    )

    return x_count.scatter_add_(-1, x_count_child, x_is_room_forward.long())
