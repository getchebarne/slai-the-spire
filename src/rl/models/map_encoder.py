import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from slai import MAP_HEIGHT
from slai import MAP_WIDTH

from src.rl.batches import get_plan_unique
from src.rl.batches import get_unique_rep_row
from src.rl.constants import MIN_BATCH_DEDUP
from src.rl.encoding import EDGE_DELTAS
from src.rl.encoding import NUM_CHANNELS
from src.rl.encoding import NUM_FEATURE_CHANNELS
from src.rl.encoding import OFF_MAP_POSITION
from src.rl.encoding import OFF_MAP_SELECTABLE
from src.rl.encoding import SLICE_MAP_EDGES
from src.rl.encoding import SLICE_MAP_ROOM_KIND
from src.rl.types import MapCache
from src.rl.types import MapDedup
from src.rl.types import MapPlan
from src.rl.types import Rollout


_NUM_CELLS = MAP_HEIGHT * MAP_WIDTH

# The dedup's packed rows: the grid's bits, 63 to an int64 word
_GRID_BITS = _NUM_CELLS * NUM_FEATURE_CHANNELS
_WORD_BITS = 63                                   # An int64 w/o its sign bit so every word is >= 0
_NUM_WORDS = math.ceil(_GRID_BITS / _WORD_BITS)   # Number of words per grid
_PAD_BITS = _NUM_WORDS * _WORD_BITS - _GRID_BITS  # Zero bits that fill the last word


class MapGNN(nn.Module):
    """Message passing over the act's Map.

    Each layer, a Room reads the mean of its children in the next row, so `num_layers` layers see
    that many rows ahead. Only the forward Rooms take part: the Rooms above the position row, or
    every Room when there is no position bit (before the act's first Room, and in the boss and
    boss treasure Rooms).

    Returns the selectable Rooms' cell features (N_rooms, dim_map), in the Room tokens' order;
    then the readout (B, dim_map): the mean over the forward Rooms, zero when there is none.

    Parameters
    ----------
    num_layers : int
        The number of message-passing layers.
    dim_map : int
        The width of the cell features and the readout.
    """

    _x_edge_delta: torch.Tensor

    def __init__(self, num_layers: int, dim_map: int) -> None:
        super().__init__()
        self._in_proj = nn.Linear(NUM_FEATURE_CHANNELS, dim_map)
        self._layers = nn.ModuleList(nn.Linear(dim_map, dim_map) for _ in range(num_layers))
        self._norms = nn.ModuleList(nn.LayerNorm(dim_map) for _ in range(num_layers))

        self.register_buffer("_x_edge_delta", torch.tensor(EDGE_DELTAS), persistent=False)

        # The GNN's computation as a function; `compile_hot_paths` swaps in its compiled version
        self.message_passing = message_passing

    def forward(
        self,
        x_grid: torch.Tensor,
        num_rooms: int,
        map_cache: MapCache | None = None,
        map_dedup: MapDedup | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the message passing over `x_grid`'s Maps.

        Parameters
        ----------
        x_grid : torch.Tensor
            (B, H, W, C) uint8. Each state's Map.
        num_rooms : int
            The number of selectable Rooms over the batch, N_rooms.
        map_cache : MapCache | None, default None
            The outputs of this rollout's last step, to reuse and update, or None to run them all.
        map_dedup : MapDedup | None, default None
            The Maps to run without a cache, planned on the host, or None to run every state's.

        Returns
        -------
        x_room_hidden : torch.Tensor
            (N_rooms, dim_map). The selectable Rooms' cell features, state after state and each
            state's in cell order: the Room tokens' order.
        x_map_readout : torch.Tensor
            (B, dim_map). Each state's readout.
        """
        batch_size = x_grid.shape[0]

        # Flatten grid from (B, H, W, C) -> (B, H * W, C)
        x_cell = x_grid.reshape(batch_size, _NUM_CELLS, NUM_CHANNELS)
        x_cell_feature = x_cell[:, :, :NUM_FEATURE_CHANNELS]  # (B, H * W, NUM_FEATURE_CHANNELS)
        if map_cache is None:
            x_hidden, x_state_map, x_readout = self._encode_maps(x_cell_feature, map_dedup)
        else:
            x_hidden, x_readout = self._encode_maps_cached(x_cell_feature, map_cache)
            x_state_map = torch.arange(batch_size, device=x_grid.device)

        # The selectable Rooms' cells, state after state and each state's in cell order
        x_room_state, x_room_cell = get_room_cells(x_grid, num_rooms)
        x_room_hidden = x_hidden[x_state_map[x_room_state], x_room_cell]
        x_map_readout = x_readout[x_state_map]

        return x_room_hidden, x_map_readout

    def _encode_maps(
        self, x_cell: torch.Tensor, map_dedup: MapDedup | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the message passing over the Maps, each distinct one once when `map_dedup` says so.

        Parameters
        ----------
        x_cell : torch.Tensor
            (B, H * W, NUM_FEATURE_CHANNELS) uint8. Each state's Map, flattened to cells.
        map_dedup : MapDedup | None
            The Maps to run and their forward Rooms by child count, or None to run every state's.

        Returns
        -------
        x_hidden : torch.Tensor
            (U, H * W, dim_map). The cell features of each Map the GNN ran on: `map_dedup`'s
            Maps, or every state's without it.
        x_inv : torch.Tensor
            (B,) long. Each state's Map, its index in `x_hidden`.
        x_readout : torch.Tensor
            (U, dim_map). The readout of each Map the GNN ran on.
        """
        batch_size = x_cell.shape[0]

        np_count_room_forward: np.ndarray | None = None
        if map_dedup is None:
            x_rep = torch.arange(batch_size, device=x_cell.device)  # Every state's own Map
            x_inv = x_rep
        else:
            # The game sits on one Room for many steps, so its grid repeats across a minibatch:
            # run the GNN on unique grids once and gather back (exact)
            x_rep, x_inv = map_dedup.x_rep, map_dedup.x_state_map
            np_count_room_forward = map_dedup.np_count_room_forward

        # Encode the unique Maps
        x_hidden, x_readout = self._encode_rooms_forward(x_cell[x_rep], np_count_room_forward)

        return x_hidden, x_inv, x_readout

    def _encode_maps_cached(
        self, x_cell: torch.Tensor, map_cache: MapCache
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rerun the message passing on the games whose Map changed since the cache's last step.

        Parameters
        ----------
        x_cell : torch.Tensor
            (E, H * W, NUM_FEATURE_CHANNELS) uint8. Each game's Map, flattened to cells.
        map_cache : MapCache
            The outputs of this rollout's last step, updated in place.

        Returns
        -------
        x_hidden : torch.Tensor
            (E, H * W, dim_map). Each game's cell features.
        x_readout : torch.Tensor
            (E, dim_map). Each game's readout.
        """
        # A new cache holds empty Maps, whose features and readout the GNN leaves at zero
        x_hidden, x_readout = map_cache.x_hidden, map_cache.x_readout
        if map_cache.x_cell is None or x_hidden is None or x_readout is None:
            map_cache.x_cell = torch.zeros_like(x_cell)
            x_hidden, x_readout = self._encode_rooms_forward(map_cache.x_cell)
            map_cache.x_hidden, map_cache.x_readout = x_hidden, x_readout

        x_changed = (x_cell != map_cache.x_cell).flatten(1).any(1)  # (E,)
        x_idx_state = torch.nonzero(x_changed).squeeze(1)
        x_hidden[x_idx_state], x_readout[x_idx_state] = self._encode_rooms_forward(
            x_cell[x_idx_state]
        )
        map_cache.x_cell = x_cell

        return x_hidden, x_readout

    def _encode_rooms_forward(
        self, x_cell: torch.Tensor, np_count_room_forward: np.ndarray | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the message passing over the Maps' forward Rooms, and put them back on the grid.

        A cell is a position along the U Maps' grids flattened together (U * H * W); an index is a
        position in the list of the N forward Rooms, most children first: the order message
        passing runs in.

        Parameters
        ----------
        x_cell : torch.Tensor
            (U, H * W, C) uint8. The Maps.
        np_count_room_forward : np.ndarray | None, default None
            (4,) int64. The Maps' forward Rooms with 0 to 3 children, as their `MapDedup` counted
            them, or None to count them here.

        Returns
        -------
        x_hidden : torch.Tensor
            (U, H * W, dim_map). Each cell's features, zero off the forward Rooms.
        x_readout : torch.Tensor
            (U, dim_map). The readout: each Map's mean over its forward Rooms, zero when it has
            none.
        """
        num_maps = x_cell.shape[0]
        num_cells = num_maps * _NUM_CELLS
        dim_map = self._in_proj.out_features
        x_cell = x_cell.float()
        x_is_room_forward = _is_room_forward(x_cell).flatten()       # (U * H * W,)
        x_cell = x_cell.flatten(0, 1)                                # (U * H * W, C)
        x_count_child = _count_children(x_cell)                      # (U * H * W,)

        # The forward Rooms with 0 to 3 children, in one host read
        if np_count_room_forward is None:
            np_count_room_forward = (
                _count_rooms_forward(x_is_room_forward, x_count_child).cpu().numpy()
            )
        num_0, num_1, num_2, num_3 = np_count_room_forward.tolist()
        num_rooms_forward = num_0 + num_1 + num_2 + num_3

        # No forward Room on any Map: every feature and readout is zero
        if num_rooms_forward == 0:
            dtype = self._in_proj.weight.dtype
            x_hidden = torch.zeros(
                num_maps, _NUM_CELLS, dim_map, dtype=dtype, device=x_cell.device
            )
            x_readout = torch.zeros(num_maps, dim_map, dtype=dtype, device=x_cell.device)

            return x_hidden, x_readout

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

        # (N, dim_map) Each forward Room's features after the message passing
        x_hidden_room_forward = self.message_passing(self, x_room_forward, x_children_by_count)

        # The forward Rooms' features on their cells, zero elsewhere
        x_hidden = torch.zeros(
            num_cells, dim_map, dtype=x_hidden_room_forward.dtype, device=x_cell.device
        )  # (U * H * W, dim_map)
        x_hidden[x_room_forward_cell] = x_hidden_room_forward
        x_hidden = x_hidden.view(num_maps, _NUM_CELLS, dim_map)  # (U, H * W, dim_map)

        # Map readout: masked mean over the forward Rooms
        x_denom = (
            x_is_room_forward.view(num_maps, _NUM_CELLS).sum(1, keepdim=True).clamp(min=1).float()
        )
        x_readout = x_hidden.sum(1) / x_denom  # Already zero off x_is_room_forward

        return x_hidden, x_readout


def message_passing(
    gnn: MapGNN, x_room_forward: torch.Tensor, x_children_by_count: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    """Run `gnn` over the forward Rooms, each Room reading the mean of its children.

    Parameters
    ----------
    gnn : MapGNN
        The GNN to run.
    x_room_forward : torch.Tensor
        (N, C) float32. The forward Rooms' cells, most children first.
    x_children_by_count : tuple[torch.Tensor, ...]
        (Nk, k) long, one per child count k, most children first. The next Nk Rooms' children, by
        index in `x_room_forward`. The Rooms past them have no children.

    Returns
    -------
    torch.Tensor
        (N, dim_map). Each Room's features.
    """
    x_hidden = gnn._in_proj(x_room_forward)  # (N, dim_map)

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


def plan_maps(rollout: Rollout) -> MapPlan:
    """Find an update's unique Maps once, over its rollout's states and final states.

    One unique and one host read for the whole update: each of its forwards then takes its
    `MapDedup` from the plan on the host, with `dedup_maps`.

    Parameters
    ----------
    rollout : Rollout
        The rollout the update trains on.

    Returns
    -------
    MapPlan
        Each state's Map, the rollout's states then its final states, and each Map's forward
        Rooms with 0 to 3 children.
    """
    # The rollout's Maps, then its final states', flattened to their feature cells
    x_grid = torch.cat([rollout.states.x_map_grid, rollout.states_final.x_map_grid])
    num_states = x_grid.shape[0]
    x_cell = x_grid.reshape(num_states, _NUM_CELLS, NUM_CHANNELS)[:, :, :NUM_FEATURE_CHANNELS]
    x_packed = _pack_rows(x_cell)  # (B + E, words)

    # x_uniq: (U, words)
    # x_inv: (B + E,) / x_inv[b] ∈ [0, U). Maps each state to its unique ID
    x_uniq, x_inv = torch.unique(x_packed, dim=0, return_inverse=True)
    x_rep = get_unique_rep_row(x_inv, x_uniq.shape[0])  # (U,)

    # Each unique Map's forward Rooms with 0 to 3 children
    x_cell_rep = x_cell[x_rep].float()  # (U, H * W, NUM_FEATURE_CHANNELS)
    x_count_room_forward = _count_rooms_forward(
        _is_room_forward(x_cell_rep), _count_children(x_cell_rep)
    )  # (U, 4)

    # The states' Maps and the counts in one host read
    np_buffer = torch.cat([x_inv, x_count_room_forward.flatten()]).cpu().numpy()

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

    # The forward's unique Maps in the plan's order, each one's first state and each state's Map
    np_map, np_rep, np_state_map = get_plan_unique(
        np_state_map_plan, len(map_plan.np_count_room_forward)
    )
    num_maps = len(np_map)

    # Both index arrays in one copy to the device, from pinned memory without blocking on CUDA
    x_buffer_cpu = torch.from_numpy(np.concatenate([np_rep, np_state_map]))
    x_buffer = (x_buffer_cpu.pin_memory() if device.type == "cuda" else x_buffer_cpu).to(
        device, non_blocking=True
    )

    return MapDedup(
        x_rep=x_buffer[:num_maps],
        x_state_map=x_buffer[num_maps:],
        np_count_room_forward=map_plan.np_count_room_forward[np_map].sum(0),
    )


def get_room_cells(x_grid: torch.Tensor, num_rooms: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Find each Room token's state and cell, in the Room tokens' order.

    The Room tokens' cells are the selectable Rooms', state after state and each state's in cell
    order.

    Parameters
    ----------
    x_grid : torch.Tensor
        (B, H, W, C) uint8. Each state's Map.
    num_rooms : int
        The number of selectable Rooms over the batch, N_rooms.

    Returns
    -------
    x_room_state : torch.Tensor
        (N_rooms,) long. Each Room token's state.
    x_room_cell : torch.Tensor
        (N_rooms,) long. Each Room token's cell on its state's grid, flattened to H * W.
    """
    x_selectable = x_grid[..., OFF_MAP_SELECTABLE].reshape(-1)  # (B * H * W,)
    x_room = torch.nonzero_static(x_selectable, size=num_rooms).squeeze(1)  # (N_rooms,)

    return x_room // _NUM_CELLS, x_room % _NUM_CELLS


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


def _pack_rows(x_cell: torch.Tensor) -> torch.Tensor:
    """Pack each grid into int64 words that sort as the grid does.

    This way unique() groups and orders them alike at a fraction of the cost: the grid's bits most
    significant first, every word non-negative.

    Parameters
    ----------
    x_cell : torch.Tensor
        (B, H * W, C) uint8. Each state's Map grid, flattened to cells.

    Returns
    -------
    torch.Tensor
        (B, words) long. Each grid's bits, 63 to a word.
    """
    batch_size = x_cell.shape[0]
    device = x_cell.device
    x_bit_shift = torch.arange(_WORD_BITS - 1, -1, -1, device=device)  # _WORD_BITS - 1 down to 0
    x_bit = F.pad(x_cell.reshape(batch_size, _GRID_BITS).long(), (0, _PAD_BITS))

    return (x_bit.view(batch_size, _NUM_WORDS, _WORD_BITS) << x_bit_shift).sum(-1)  # (B, words)
