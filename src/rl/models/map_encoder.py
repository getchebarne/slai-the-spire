import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.constants import DEDUP_MIN_BATCH
from src.rl.constants import MAP_HEIGHT
from src.rl.constants import MAP_WIDTH
from src.rl.encoding import EDGE_DELTAS
from src.rl.encoding import NUM_CHANNELS
from src.rl.encoding import NUM_EDGE_CHANNELS
from src.rl.encoding import NUM_FEATURE_CHANNELS
from src.rl.encoding import OFF_MAP_POSITION
from src.rl.encoding import OFF_MAP_SELECTABLE
from src.rl.encoding import SLICE_MAP_EDGES
from src.rl.encoding import SLICE_MAP_ROOM_KIND
from src.rl.types import MapOutput


_NUM_NODES = MAP_HEIGHT * MAP_WIDTH

# The dedup's packed rows: the grid's 0/1 cells 63 to an int64 word
_GRID_BITS = _NUM_NODES * NUM_FEATURE_CHANNELS


def _build_idx_child() -> torch.Tensor:
    x_idx_child = torch.full((_NUM_NODES, NUM_EDGE_CHANNELS), _NUM_NODES, dtype=torch.long)
    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            idx_node = y * MAP_WIDTH + x  # Flat index
            if y + 1 < MAP_HEIGHT:
                for idx_edge, delta in enumerate(EDGE_DELTAS):
                    x_next = x + delta  # {-1, 0, +1}
                    if 0 <= x_next < MAP_WIDTH:
                        x_idx_child[idx_node, idx_edge] = (y + 1) * MAP_WIDTH + x_next
    return x_idx_child


def _pack_rows(x_node: torch.Tensor) -> torch.Tensor:
    """Pack each grid into int64 words that sort as the grid does.

    This way unique() groups and orders them alike at a fraction of the cost: the 0/1 cells most
    significant first, every word non-negative.

    Parameters
    ----------
    x_node : torch.Tensor
        (B, H * W, C) uint8. Each state's Map grid, flattened to nodes.

    Returns
    -------
    torch.Tensor
        (B, words) long. Each grid's 0/1 cells, 63 to a word.
    """
    batch_size, device = x_node.shape[0], x_node.device
    x_bit_shift = torch.arange(62, -1, -1, device=device)
    x_bit = F.pad(x_node.reshape(batch_size, -1).long(), (0, -_GRID_BITS % 63))
    return (x_bit.view(batch_size, -1, 63) << x_bit_shift).sum(-1)  # (B, words)


class MapGNN(nn.Module):
    """Message passing over the act's Map.

    Each layer, a Room reads the mean of its children in the next row, so `num_layers` layers see
    that many rows ahead. Only the Rooms above the position row take part, or every Room when
    there is no position bit (before the act's first Room, and in the boss and boss treasure
    Rooms).

    Returns the selectable Rooms' node features (N_rooms, hidden), in the Room tokens' order;
    then the readout (B, hidden): the mean over the Rooms that take part, zero when none does.

    Parameters
    ----------
    num_layers : int
        The number of message-passing layers.
    hidden : int
        The width of the node features and the readout.
    """

    _x_idx_child: torch.Tensor

    def __init__(self, num_layers: int, hidden: int) -> None:
        super().__init__()
        self._in_proj = nn.Linear(NUM_FEATURE_CHANNELS, hidden)
        self._layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(num_layers))
        self._norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(num_layers))

        self.register_buffer("_x_idx_child", _build_idx_child(), persistent=False)
        self.register_buffer(
            "_x_row_of_node", (torch.arange(_NUM_NODES) // MAP_WIDTH).float(), persistent=False
        )

        # The GNN's computation as a plain function, which actor_critic.compile_hot_paths swaps for
        # its compiled version on the Learner's model
        self.message_passing = message_passing

    def forward(
        self, x_grid: torch.Tensor, num_rooms: int, map_output: MapOutput | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the message passing over `x_grid`'s Maps.

        Parameters
        ----------
        x_grid : torch.Tensor
            (B, H, W, C) uint8. Each state's Map.
        num_rooms : int
            The number of selectable Rooms over the batch, N_rooms.
        map_output : MapOutput | None, default None
            The outputs of this rollout's last step, to reuse and update, or None to run them all.

        Returns
        -------
        x_room_hidden : torch.Tensor
            (N_rooms, hidden). The selectable Rooms' node features, state after state and each
            state's in node order: the Room tokens' order.
        x_map_readout : torch.Tensor
            (B, hidden). Each state's readout. With `map_output`, this is its tensor, which the
            next call overwrites.
        """
        batch_size = x_grid.shape[0]

        # Flatten grid from (B, H, W, C) -> (B, H * W, C)
        x_node = x_grid.reshape(batch_size, _NUM_NODES, NUM_CHANNELS)
        x_hidden_map, x_map_of_state, x_map_readout = self._encode_maps(
            x_node[:, :, :NUM_FEATURE_CHANNELS], map_output
        )

        # The selectable Rooms' nodes, state after state and each state's in node order
        x_selectable = x_node[:, :, OFF_MAP_SELECTABLE].reshape(-1)  # (B * H * W,)
        x_room = torch.nonzero_static(x_selectable, size=num_rooms).squeeze(1)  # (N_rooms,)
        x_room_state = x_room // _NUM_NODES
        x_room_node = x_room % _NUM_NODES
        return x_hidden_map[x_map_of_state[x_room_state], x_room_node], x_map_readout

    def _encode_maps(
        self, x_node: torch.Tensor, map_output: MapOutput | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the message passing over the Maps, each distinct one once on a large batch.

        Parameters
        ----------
        x_node : torch.Tensor
            (B, H * W, NUM_FEATURE_CHANNELS) uint8. Each state's Map, flattened to nodes.
        map_output : MapOutput | None
            The outputs of this rollout's last step, to reuse and update, or None to run them all.

        Returns
        -------
        x_hidden_map : torch.Tensor
            (U, H * W, hidden). The node features of each Map the GNN ran on: the unique Maps
            on a batch of DEDUP_MIN_BATCH states or more, every state's on a smaller one or with
            `map_output`. With it, this is its tensor, which the next call overwrites.
        x_map_of_state : torch.Tensor
            (B,) long. Each state's Map, its index in `x_hidden_map`.
        x_map_readout : torch.Tensor
            (B, hidden). Each state's readout. With `map_output`, this is its tensor too.
        """
        batch_size = x_node.shape[0]

        if map_output is not None:
            x_hidden, x_readout = map_output.x_hidden, map_output.x_readout
            if map_output.x_node is None or x_hidden is None or x_readout is None:
                x_hidden, x_readout = self._encode_forward_rooms(x_node)
                map_output.x_hidden, map_output.x_readout = x_hidden, x_readout
            else:
                x_changed = (x_node != map_output.x_node).flatten(1).any(1)  # (B,)
                x_idx_state = torch.nonzero(x_changed).squeeze(1)
                if x_idx_state.shape[0] > 0:
                    x_hidden[x_idx_state], x_readout[x_idx_state] = self._encode_forward_rooms(
                        x_node[x_idx_state]
                    )
            map_output.x_node = x_node
            return x_hidden, torch.arange(batch_size, device=x_node.device), x_readout

        if batch_size < DEDUP_MIN_BATCH:
            x_hidden, x_readout = self._encode_forward_rooms(x_node)
            return x_hidden, torch.arange(batch_size, device=x_node.device), x_readout

        # The game sits on one Map node for many steps, so its grid repeats across a minibatch:
        # run the GNN on unique grids once and gather back (exact)
        x_packed = _pack_rows(x_node)  # (B, words)

        # x_uniq: (U, words)
        # x_inverse: (B,) / x_inverse[b] ∈ [0, U). Maps each state to its unique ID
        x_uniq, x_inverse = torch.unique(x_packed, dim=0, return_inverse=True)

        x_rep = torch.zeros(x_uniq.shape[0], dtype=torch.long, device=x_node.device)
        x_rep[x_inverse] = torch.arange(batch_size, device=x_node.device)

        # Encode the unique Maps, and gather the readout back
        x_hidden, x_readout_uniq = self._encode_forward_rooms(x_node[x_rep])
        return x_hidden, x_inverse, x_readout_uniq[x_inverse]

    def _encode_forward_rooms(self, x_node: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the message passing over the Maps' forward Rooms, and put them back on the grid.

        Parameters
        ----------
        x_node : torch.Tensor
            (U, H * W, C) uint8. The Maps.

        Returns
        -------
        x_hidden : torch.Tensor
            (U, H * W, hidden). Each node's features, zero off the forward Rooms.
        x_readout : torch.Tensor
            (U, hidden). The readout: each Map's mean over its forward Rooms, zero when it has
            none.
        """
        x_node = x_node.float()
        x_valid = self._forward_valid(x_node)  # (U, H * W) / Forward Rooms only
        x_num_children = x_node[:, :, SLICE_MAP_EDGES].sum(-1).long()  # (U, H * W)

        # The forward Rooms of each child count, in the one read: 0 counts the other nodes
        x_key = ((x_num_children + 1) * x_valid).flatten()
        x_count = torch.zeros(NUM_EDGE_CHANNELS + 2, dtype=torch.long, device=x_node.device)
        _, num_0, num_1, num_2, num_3 = x_count.index_add_(
            0, x_key, torch.ones_like(x_key)
        ).tolist()

        # x_room_map, x_room_node: (N,) / Each of the N forward Rooms' Map and node, most
        # children first
        x_room_map_node = torch.nonzero_static(x_valid, size=num_0 + num_1 + num_2 + num_3)
        x_room_map, x_room_node = x_room_map_node.unbind(1)
        x_order = torch.argsort(x_num_children[x_room_map, x_room_node], descending=True)
        x_room_map, x_room_node = x_room_map[x_order], x_room_node[x_order]
        x_room = x_node[x_room_map, x_room_node]  # (N, C)

        # Each forward Room's index in `x_room`
        x_room_of_node = torch.zeros(x_valid.shape, dtype=torch.long, device=x_node.device)
        x_room_of_node[x_room_map, x_room_node] = torch.arange(
            x_room_map.shape[0], device=x_node.device
        )

        # x_child: (E,) / Each edge's child, a forward Room too, in the Rooms' order
        x_edge_room, x_idx_edge = torch.nonzero_static(
            x_room[:, SLICE_MAP_EDGES], size=num_1 + 2 * num_2 + 3 * num_3
        ).unbind(1)
        x_child_node = self._x_idx_child[x_room_node[x_edge_room], x_idx_edge]
        x_child = x_room_of_node[x_room_map[x_edge_room], x_child_node]

        # One block of Rooms per child count
        x_3, x_2, x_1 = x_child.split([3 * num_3, 2 * num_2, num_1])
        x_children_by_count = (x_3.view(num_3, 3), x_2.view(num_2, 2), x_1.view(num_1, 1))

        x_hidden_room = self.message_passing(self, x_room, x_children_by_count)
        x_hidden = torch.zeros(
            *x_valid.shape, x_hidden_room.shape[1], dtype=x_node.dtype, device=x_node.device
        )
        x_hidden[x_room_map, x_room_node] = x_hidden_room

        # Map readout: masked mean over valid nodes
        x_denom = x_valid.sum(1, keepdim=True).clamp(min=1).float()
        x_readout = x_hidden.sum(1) / x_denom  # Already zero off x_valid

        return x_hidden, x_readout

    def _forward_valid(self, x_node: torch.Tensor) -> torch.Tensor:
        """Compute the valid Rooms matrix (higher than current floor and not None).

        Parameters
        ----------
        x_node : torch.Tensor
            (U, H * W, C) float32. Each Map's nodes.

        Returns
        -------
        torch.Tensor
            (U, H * W) bool. Whether each node is a valid Room.
        """

        # The encoder sets one Room-kind bit per Room, none elsewhere, and at most one
        # position bit, on the current Room
        x_room_exists = x_node[:, :, SLICE_MAP_ROOM_KIND].sum(-1) > 0  # (U, H * W)
        x_position = x_node[:, :, OFF_MAP_POSITION]  # Current-position bit: (U, H * W)
        x_row_current = (x_position * self._x_row_of_node).sum(
            -1
        )  # (U,) / Current floor, 0 if none
        x_row_current = torch.where(
            condition=x_position.sum(-1) > 0,  # (U,)
            input=x_row_current,  # (U,)
            other=torch.full(
                (), -1.0, dtype=x_row_current.dtype, device=x_row_current.device
            ),  # (U,)
        )
        x_forward = self._x_row_of_node.unsqueeze(0) > x_row_current.unsqueeze(1)  # (U, H * W)
        return x_room_exists & x_forward  # (U, H * W)


def message_passing(
    gnn: MapGNN, x_room: torch.Tensor, x_children_by_count: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    """Run `gnn` over the forward Rooms, each Room reading the mean of its children.

    Parameters
    ----------
    gnn : MapGNN
        The GNN to run.
    x_room : torch.Tensor
        (N, C) float32. The forward Rooms' nodes, most children first.
    x_children_by_count : tuple[torch.Tensor, ...]
        (Nk, k) long, one per child count k. The next Nk Rooms' children, by index in `x_room`.
        The Rooms past them have no children.

    Returns
    -------
    torch.Tensor
        (N, hidden). Each Room's features.
    """
    x_hidden = gnn._in_proj(x_room)
    for lin, norm in zip(gnn._layers, gnn._norms):
        x_mean = torch.cat([x_hidden[x_child].mean(1) for x_child in x_children_by_count])
        x_mean = F.pad(x_mean, (0, 0, 0, x_hidden.shape[0] - x_mean.shape[0]))  # Zero if childless
        x_hidden = norm(x_hidden + F.relu(lin(x_mean)))

    return x_hidden
