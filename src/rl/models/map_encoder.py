import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from slai import MAP_HEIGHT
from slai import MAP_WIDTH

from src.rl.batches import dedup_rows
from src.rl.batches import get_unique_rep_row
from src.rl.constants import EDGE_DELTAS
from src.rl.constants import MIN_BATCH_DEDUP
from src.rl.encoding import ENCODING_DIM_ROOM
from src.rl.encoding import OFF_MAP_ROOM_FORWARD
from src.rl.encoding import OFF_MAP_SELECTABLE
from src.rl.encoding import SLICE_MAP_EDGES
from src.rl.models.entity_projector import mix_hash
from src.rl.types import CollectionKind
from src.rl.types import Live
from src.rl.types import MapCache
from src.rl.types import MapDedup
from src.rl.types import MapPlan
from src.rl.types import States


_NUM_CELLS = MAP_HEIGHT * MAP_WIDTH

# The Map hash's odd weights, from a generator of their own
_HASH_WEIGHT_CELL, _HASH_WEIGHT_CHANNELS = (
    torch.randint(1 << 62, (2,), generator=torch.Generator().manual_seed(1)) | 1
).tolist()


class MapGNN(nn.Module):
    """The Rooms' projector: a projection of each Room's features, then message passing.

    The projection has the Entity projector's shape and weights of its own: a first layer, a
    second layer and a LayerNorm. Each message-passing layer, a Room reads the mean of its
    children in the next row, so `num_layers` layers see that many rows ahead. Only the forward
    Rooms take part.

    The Map's topology alone says which cells hold a forward Room, where its edges lead and
    whether it is selectable. The Map's features are read whole, by the first layer.

    Returns the selectable Rooms' rows (N_rooms, dim_entity), in the Room tokens' order; then
    the readout (B, dim_map): the mean over the forward Rooms, zero when there is none.

    Parameters
    ----------
    num_layers : int
        The number of message-passing layers.
    dim_map : int
        The width of the projection, the cell features and the readout.
    dim_entity : int
        The width of the selectable Rooms' rows: the tokens'.
    """

    _x_edge_delta: torch.Tensor

    def __init__(self, num_layers: int, dim_map: int, dim_entity: int) -> None:
        super().__init__()

        # The Rooms' projection: a first layer, a second layer (after ReLU) and a LayerNorm
        self._pre_room = nn.Linear(ENCODING_DIM_ROOM, dim_map)
        self._post_room = nn.Linear(dim_map, dim_map)
        self._layer_norm_room = nn.LayerNorm(dim_map)

        self._in_proj = nn.Linear(dim_map, dim_map)
        self._layers = nn.ModuleList(nn.Linear(dim_map, dim_map) for _ in range(num_layers))
        self._norms = nn.ModuleList(nn.LayerNorm(dim_map) for _ in range(num_layers))
        self._out_proj = nn.Identity() if dim_map == dim_entity else nn.Linear(dim_map, dim_entity)

        self.register_buffer("_x_edge_delta", torch.tensor(EDGE_DELTAS), persistent=False)

        # The GNN's computation as a function; `compile_hot_paths` swaps in its compiled version
        self.message_passing = message_passing

    def forward(
        self,
        states: States[Live],
        map_cache: MapCache | None = None,
        map_dedup: MapDedup | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the GNN over `states`' Maps.

        Parameters
        ----------
        states : States[Live]
            The game states: their Maps' features and topology.
        map_cache : MapCache | None, default None
            The outputs of this rollout's last step, to reuse and update, or None to run them all.
        map_dedup : MapDedup | None, default None
            The Maps to run without a cache, planned on the host, or None to run every state's.

        Returns
        -------
        x_room_hidden : torch.Tensor
            (N_rooms, dim_entity). The selectable Rooms' rows, state after state and each state's
            in cell order: the Room tokens' order.
        x_map_readout : torch.Tensor
            (B, dim_map). Each state's readout.
        """
        batch_size = states.x_map_topology.shape[0]
        x_feature = states.x_map_feature.flatten(1, 2)  # (B, H * W, ENCODING_DIM_ROOM)
        x_topology = states.x_map_topology.flatten(1, 2)  # (B, H * W, C)

        if map_cache is None:
            x_hidden, x_state_map, x_readout = self._encode_maps(x_feature, x_topology, map_dedup)
        else:
            x_hidden, x_readout = self._encode_maps_cached(x_feature, x_topology, map_cache)
            x_state_map = torch.arange(batch_size, device=x_topology.device)

        # The selectable Rooms' cells, state after state and each state's in cell order
        num_rooms = int(states.x_total_cpu[:, CollectionKind.ROOMS].sum())
        x_is_room_selectable = x_topology[:, :, OFF_MAP_SELECTABLE] != 0  # (B, H * W)
        x_room = torch.nonzero_static(x_is_room_selectable.flatten(), size=num_rooms).squeeze(1)
        x_room_state, x_room_cell = x_room // _NUM_CELLS, x_room % _NUM_CELLS  # (N_rooms,)
        x_room_hidden = self._out_proj(x_hidden[x_state_map[x_room_state], x_room_cell])
        x_map_readout = x_readout[x_state_map]

        return x_room_hidden, x_map_readout

    def _encode_maps(
        self, x_feature: torch.Tensor, x_topology: torch.Tensor, map_dedup: MapDedup | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the GNN over the Maps, each distinct one once when `map_dedup` says so.

        Parameters
        ----------
        x_feature : torch.Tensor
            (B, H * W, ENCODING_DIM_ROOM) uint8. Each state's Map features, flattened to cells.
        x_topology : torch.Tensor
            (B, H * W, C) uint8. Each state's Map topology, flattened to cells.
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
        batch_size = x_topology.shape[0]

        np_count_room_forward: np.ndarray | None = None
        if map_dedup is None:
            x_rep = torch.arange(batch_size, device=x_topology.device)  # Every state's own Map
            x_inv = x_rep
        else:
            # Each unique Map runs once and gathers back, up to a hash collision
            x_rep, x_inv = map_dedup.x_rep, map_dedup.x_state_map
            np_count_room_forward = map_dedup.np_count_room_forward

        # Encode the unique Maps
        x_hidden, x_readout = self._encode_rooms_forward(
            x_feature[x_rep], x_topology[x_rep], np_count_room_forward
        )

        return x_hidden, x_inv, x_readout

    def _encode_maps_cached(
        self, x_feature: torch.Tensor, x_topology: torch.Tensor, map_cache: MapCache
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rerun the GNN on the games whose Map changed since the cache's last step.

        Parameters
        ----------
        x_feature : torch.Tensor
            (E, H * W, ENCODING_DIM_ROOM) uint8. Each game's Map features, flattened to cells.
        x_topology : torch.Tensor
            (E, H * W, C) uint8. Each game's Map topology, flattened to cells.
        map_cache : MapCache
            The outputs of this rollout's last step, updated in place.

        Returns
        -------
        x_hidden : torch.Tensor
            (E, H * W, dim_map). Each game's cell features.
        x_readout : torch.Tensor
            (E, dim_map). Each game's readout.
        """
        x_hash_map = _hash_maps(x_feature, x_topology)  # (E,)

        # A new cache runs every game's Map; then only the Maps whose hash changed
        x_hidden, x_readout = map_cache.x_hidden, map_cache.x_readout
        if map_cache.x_hash_map is None or x_hidden is None or x_readout is None:
            x_hidden, x_readout = self._encode_rooms_forward(x_feature, x_topology)
            map_cache.x_hidden, map_cache.x_readout = x_hidden, x_readout
        else:
            x_changed = x_hash_map != map_cache.x_hash_map  # (E,)
            x_idx_state = torch.nonzero(x_changed).squeeze(1)
            x_hidden[x_idx_state], x_readout[x_idx_state] = self._encode_rooms_forward(
                x_feature[x_idx_state], x_topology[x_idx_state]
            )

        map_cache.x_hash_map = x_hash_map

        return x_hidden, x_readout

    def _encode_rooms_forward(
        self,
        x_feature: torch.Tensor,
        x_topology: torch.Tensor,
        np_count_room_forward: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the GNN over the Maps' forward Rooms, and put them back on the grid.

        A cell is a position along the U Maps' grids flattened together (U * H * W); an index is a
        position in the list of the N forward Rooms, most children first: the order message
        passing runs in.

        Parameters
        ----------
        x_feature : torch.Tensor
            (U, H * W, ENCODING_DIM_ROOM) uint8. The Maps' features.
        x_topology : torch.Tensor
            (U, H * W, C) uint8. The Maps' topology.
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
        num_maps = x_topology.shape[0]
        num_cells = num_maps * _NUM_CELLS
        dim_map = self._in_proj.out_features
        x_feature = x_feature.flatten(0, 1)               # (U * H * W, ENCODING_DIM_ROOM)
        x_topology = x_topology.flatten(0, 1)             # (U * H * W, C)
        x_is_room_forward = _is_room_forward(x_topology)  # (U * H * W,)
        x_count_child = _count_children(x_topology)       # (U * H * W,)

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
                num_maps, _NUM_CELLS, dim_map, dtype=dtype, device=x_topology.device
            )
            x_readout = torch.zeros(num_maps, dim_map, dtype=dtype, device=x_topology.device)

            return x_hidden, x_readout

        # Each forward Room's cell, most children first
        x_room_forward_cell = torch.nonzero_static(
            x_is_room_forward, size=num_rooms_forward
        ).squeeze(1)  # (N,)
        x_room_forward_cell = x_room_forward_cell[
            torch.argsort(x_count_child[x_room_forward_cell], descending=True)
        ]

        # First layer: the forward Rooms' features, read whole
        x_first = self._pre_room(x_feature[x_room_forward_cell].to(self._pre_room.weight.dtype))

        # (N, dim_map) Each forward Room's projected row: second layer + norm
        x_room_forward = self._layer_norm_room(self._post_room(F.relu(x_first)))

        # (U * H * W,) Each forward Room's index at its cell, to turn a child's cell into its index
        x_idx_room_forward = torch.zeros(num_cells, dtype=torch.long, device=x_topology.device)
        x_idx_room_forward[x_room_forward_cell] = torch.arange(
            num_rooms_forward, device=x_topology.device
        )

        # Each edge's child, by index: an edge leads to the next row, `delta` columns over
        x_edge_parent, x_idx_edge = torch.nonzero_static(
            x_topology[x_room_forward_cell][:, SLICE_MAP_EDGES] != 0,
            size=num_1 + 2 * num_2 + 3 * num_3,
        ).unbind(1)
        x_child_cell = (
            x_room_forward_cell[x_edge_parent] + MAP_WIDTH + self._x_edge_delta[x_idx_edge]
        )
        x_child = x_idx_room_forward[x_child_cell]  # (num_edges,)

        # (Nk, k) The children of the Rooms with 3, 2 and 1 child, which come first in that order
        x_3, x_2, x_1 = x_child.split([3 * num_3, 2 * num_2, num_1])
        x_children_by_count = (x_3.view(num_3, 3), x_2.view(num_2, 2), x_1.view(num_1, 1))

        # (N, dim_map) Each forward Room's features after the message passing
        x_hidden_room_forward = self.message_passing(
            self, self._in_proj(x_room_forward), x_children_by_count
        )

        # The forward Rooms' features on their cells, zero elsewhere
        x_hidden = torch.zeros(
            num_cells, dim_map, dtype=x_hidden_room_forward.dtype, device=x_topology.device
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
    gnn: MapGNN, x_hidden: torch.Tensor, x_children_by_count: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    """Run `gnn` over the forward Rooms, each Room reading the mean of its children.

    Parameters
    ----------
    gnn : MapGNN
        The GNN to run.
    x_hidden : torch.Tensor
        (N, dim_map). The forward Rooms' rows after `_in_proj`, most children first.
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


def plan_maps(states: States[Live]) -> MapPlan:
    """Find the update's unique Maps, count each one's forward Rooms and read the plan to the host.

    Parameters
    ----------
    states : States[Live]
        The rollout's states followed by its final states.

    Returns
    -------
    MapPlan
        Each state's Map, by its plan ID, and each unique Map's forward Rooms with 0 to 3 children.
    """
    x_topology = states.x_map_topology.flatten(1, 2)  # (B, H * W, C)
    x_hash_map = _hash_maps(states.x_map_feature.flatten(1, 2), x_topology)  # (B,)

    # x_uniq: (U_plan,)
    # x_state_map: (B,) / x_state_map[b] ∈ [0, U_plan). Maps each state to its Map's plan ID
    x_uniq, x_state_map = torch.unique(x_hash_map, return_inverse=True)
    x_rep = get_unique_rep_row(x_state_map, x_uniq.shape[0])  # (U_plan,)

    # Each unique Map's forward Rooms with 0 to 3 children, from its first state's topology
    x_topology_rep = x_topology[x_rep]  # (U_plan, H * W, C)
    x_count_room_forward = _count_rooms_forward(
        _is_room_forward(x_topology_rep), _count_children(x_topology_rep)
    )

    # The states' Maps and the counts in one host read
    num_states = x_state_map.shape[0]
    np_buffer = torch.cat([x_state_map, x_count_room_forward.flatten()]).cpu().numpy()

    return MapPlan(
        np_state_map=np_buffer[:num_states],
        np_count_room_forward=np_buffer[num_states:].reshape(x_count_room_forward.shape),
    )


def dedup_maps(
    map_plan: MapPlan, np_idx_state: np.ndarray, device: torch.device
) -> MapDedup | None:
    """Take one forward's `MapDedup` from its update's plan, on the host.

    A forward of MIN_BATCH_DEDUP states or more runs its unique Maps, in the plan's order, each
    from its first state; a smaller one runs every state's, without a `MapDedup`.

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
    MapDedup | None
        The Maps the forward runs, each state's among them, and their forward Rooms with 0 to 3
        children; None for a forward under MIN_BATCH_DEDUP states.
    """
    if len(np_idx_state) < MIN_BATCH_DEDUP:
        return None

    np_state_map_plan = map_plan.np_state_map[np_idx_state]  # (B,)
    x_rep, x_state_map, np_count_room_forward = dedup_rows(
        np_state_map_plan, map_plan.np_count_room_forward, device
    )

    return MapDedup(
        x_rep=x_rep, x_state_map=x_state_map, np_count_room_forward=np_count_room_forward
    )


def _is_room_forward(x_topology: torch.Tensor) -> torch.Tensor:
    """Find the forward Rooms: the cells whose topology marks one.

    Parameters
    ----------
    x_topology : torch.Tensor
        (..., C) uint8. Map cells' topology, along any leading axes.

    Returns
    -------
    torch.Tensor
        (...) bool. Whether each cell holds a forward Room.
    """
    return x_topology[..., OFF_MAP_ROOM_FORWARD] != 0


def _hash_maps(x_feature: torch.Tensor, x_topology: torch.Tensor) -> torch.Tensor:
    """Hash each Map: what the GNN reads of it.

    For each cell that holds a forward Room: the cell, its features and its edges, a bit per
    channel. Each cell's hash is mixed before the Map sums them.

    Parameters
    ----------
    x_feature : torch.Tensor
        (B, H * W, ENCODING_DIM_ROOM) uint8. Each state's Map features, flattened to cells.
    x_topology : torch.Tensor
        (B, H * W, C) uint8. Each state's Map topology, flattened to cells.

    Returns
    -------
    torch.Tensor
        (B,) long. Each Map's hash, 0 if no cell holds a forward Room.
    """
    x_is_room_forward = _is_room_forward(x_topology)  # (B, H * W)

    # Each cell's features and edges as one number, a bit per channel
    x_channel = torch.cat([x_feature, x_topology[:, :, SLICE_MAP_EDGES]], dim=2)
    x_idx_channel = torch.arange(x_channel.shape[2], device=x_channel.device)
    x_channel_mask = (x_channel.long() << x_idx_channel).sum(-1)  # (B, H * W)

    # Mix first: a Map's hash sums its cells, so two cells that swap their features would tie
    x_idx_cell = torch.arange(_NUM_CELLS, device=x_channel.device)
    x_hash_cell = mix_hash(
        (x_idx_cell + 1) * _HASH_WEIGHT_CELL + x_channel_mask * _HASH_WEIGHT_CHANNELS
    )  # (B, H * W)

    return (x_hash_cell * x_is_room_forward).sum(1)


def _count_children(x_topology: torch.Tensor) -> torch.Tensor:
    """Count each cell's children: its edges to the next row.

    Parameters
    ----------
    x_topology : torch.Tensor
        (..., C) uint8. Map cells' topology, along any leading axes.

    Returns
    -------
    torch.Tensor
        (...) long. Each cell's number of children.
    """
    return x_topology[..., SLICE_MAP_EDGES].sum(-1).long()


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
