"""The game state's rows.

Its states taken and joined, the shared step's regions written and joined, and its one copy to a
device.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F

from src.rl.types import X_EffectPending
from src.rl.types import X_GameState
from src.rl.types import X_OwnedSparse
from src.rl.types import X_Tokens


# A function on a field
type _FieldFn = Callable[[torch.Tensor], torch.Tensor]

# A block of the state, its rows held sparse
type _Block = X_Tokens | X_OwnedSparse | X_EffectPending


def take_states(
    x_gs: X_GameState, x_states: torch.Tensor, x_count_total: torch.Tensor
) -> X_GameState:
    """Take the states `x_states` of `x_gs`, in that order, packed on their own.

    It serves the minibatches and their shuffle.

    Parameters
    ----------
    x_gs : X_GameState
        The game state to take from.
    x_states : torch.Tensor
        (S,) long. The states to take.
    x_count_total : torch.Tensor
        (number of blocks,) long. The taken states' rows in each block, in `count_blocks`' order,
        summed on the CPU from the counts there: the sizes, which would otherwise be read back.

    Returns
    -------
    X_GameState
        The taken states, in `x_states`' order, packed on their own.
    """
    num_tokens, *num_rows_owned, num_effect_pending = x_count_total.tolist()
    x = x_gs
    x_count = x.tokens.x_count
    fns_owned = [
        _taken(_count_state(x_owned.x_count, x_count), x_states, num_rows)
        for x_owned, num_rows in zip(_owned(x.tokens), num_rows_owned)
    ]
    return _map_rows(
        x,
        _taken(x_count, x_states, num_tokens),
        fns_owned,
        _taken(x.effect_pending.x_count, x_states, num_effect_pending),
        lambda x_field: x_field[x_states],
    )


def count_blocks(x_gs: X_GameState) -> torch.Tensor:
    """Count each state's rows in each block, the blocks in `_blocks`' order.

    The rollout keeps the counts on the CPU, where the minibatches sum `take_states`' sizes.

    Parameters
    ----------
    x_gs : X_GameState
        The game state.

    Returns
    -------
    torch.Tensor
        (B, number of blocks) long. Each state's rows in each block.
    """
    x = x_gs
    return torch.stack(
        [
            x.tokens.x_count,
            *(_count_state(x_owned.x_count, x.tokens.x_count) for x_owned in _owned(x.tokens)),
            x.effect_pending.x_count,
        ],
        dim=1,
    )


def cat_states(x_gss: list[X_GameState]) -> X_GameState:
    """Join the states one after another, as one.

    The rollout joins its steps with it.

    Parameters
    ----------
    x_gss : list[X_GameState]
        The game states to join.

    Returns
    -------
    X_GameState
        Their states one after another, every block's entries padded to the widest of theirs.
    """
    xs = _widened(x_gss)
    x_fields = iter([torch.cat(x_fields_states) for x_fields_states in zip(*map(_fields, xs))])
    return _map_fields(xs[0], lambda x_field: next(x_fields))


def to_device(x_gs: X_GameState, device: torch.device) -> X_GameState:
    """Copy `x_gs` onto `device` in one copy.

    On the CPU, the buffer is the copy.

    Parameters
    ----------
    x_gs : X_GameState
        The game state to copy.
    device : torch.device
        The device to copy it onto.

    Returns
    -------
    X_GameState
        `x_gs` on `device`, every field a view of one buffer.
    """

    # Every field packed into one pinned buffer and sent in one copy: a copy per field from
    # pageable memory makes the CPU wait for the GPU once per field
    fields = _fields(x_gs)
    starts, end = [], 0
    for t in fields:
        starts.append(end)
        end += -(-t.numel() * t.element_size() // 8) * 8  # 8-byte aligned for every dtype
    staging = torch.empty(end, dtype=torch.uint8, pin_memory=device.type == "cuda")
    for t, start in zip(fields, starts):
        _bytes_as(staging, start, t).copy_(t)
    packed = staging.to(device, non_blocking=True) if device.type == "cuda" else staging
    pieces = iter(_bytes_as(packed, start, t) for t, start in zip(fields, starts))
    return _map_fields(x_gs, lambda x_field: next(pieces))


def share_memory(x_gs: X_GameState) -> X_GameState:
    """Move every field of `x_gs` to shared memory.

    Parameters
    ----------
    x_gs : X_GameState
        The game state.

    Returns
    -------
    X_GameState
        `x_gs` with every field moved to shared memory.
    """
    return _map_fields(x_gs, torch.Tensor.share_memory_)


def region_of_states(x_gs: X_GameState, first_state: int, num_states: int) -> X_GameState:
    """Return the region of the states [first_state, first_state + num_states) in `x_gs`.

    Parameters
    ----------
    x_gs : X_GameState
        A shared step's game state, whose every field holds the same rows per state.
    first_state : int
        The region's first state.
    num_states : int
        The region's number of states.

    Returns
    -------
    X_GameState
        The region: the states' counts and per-state fields, and their share of every block's
        rows, views all.
    """
    num_states_step = len(x_gs.tokens.x_count)

    def region(x_field: torch.Tensor) -> torch.Tensor:
        num_rows_state = len(x_field) // num_states_step
        return x_field[first_state * num_rows_state : (first_state + num_states) * num_rows_state]

    return _map_fields(x_gs, region)


def write_rows(x_region: X_GameState, x_gs: X_GameState) -> None:
    """Write `x_gs` into a shared step's region, its states'.

    Every block's rows go at the region's start, and the counts and per-state fields beside them.

    Parameters
    ----------
    x_region : X_GameState
        The shared step's region of `x_gs`'s states, written in place.
    x_gs : X_GameState
        The region's states, as the encoders write them.
    """
    for x_region_field, x_field in zip(_fields(x_region), _fields(x_gs)):
        _copy_rows(x_region_field, x_field)


def join_regions(x_gs: X_GameState, num_regions: int) -> X_GameState:
    """Join a shared step's `x_gs` of `num_regions` equal regions as one game state.

    Parameters
    ----------
    x_gs : X_GameState
        A shared step's game state, of `num_regions` equal regions.
    num_regions : int
        The number of regions.

    Returns
    -------
    X_GameState
        Every block's leading rows of each region, region after region, as its counts say, its
        sparse entries cut to the most a row uses; the states' counts and the per-state fields as
        they are.
    """
    x = x_gs
    x_count = x.tokens.x_count
    fn_token = _joined(x_count, len(x.tokens.x_token_kind), num_regions)

    # An owned row kind's states hold the rows their joined tokens' counts say
    fns_owned = [
        _joined(
            _count_state(fn_token(x_owned.x_count), x_count), len(x_owned.x_column), num_regions
        )
        for x_owned in _owned(x.tokens)
    ]
    x_joined = _map_rows(
        x,
        fn_token,
        fns_owned,
        _joined(x.effect_pending.x_count, len(x.effect_pending.x_column), num_regions),
        lambda x_field: x_field,
    )
    for block in _blocks(x_joined):
        _cut_entries(block)
    return x_joined


def _rows_of_states(
    x_first_row: torch.Tensor, x_count: torch.Tensor, num_rows: int
) -> torch.Tensor:
    """Lay out the rows of states one after another.

    Parameters
    ----------
    x_first_row : torch.Tensor
        (S,) long. Each state's first row.
    x_count : torch.Tensor
        (S,) long. Each state's number of rows.
    num_rows : int
        The rows' total; it sizes the output without a read back.

    Returns
    -------
    torch.Tensor
        (num_rows,) long. The rows, state after state: each state's `x_count` rows from its
        `x_first_row` on.
    """
    x_shift = x_first_row - (x_count.cumsum(0) - x_count)
    x_shift_row = x_shift.repeat_interleave(x_count, output_size=num_rows)
    return torch.arange(num_rows, device=x_count.device) + x_shift_row


def _count_state(x_count_token: torch.Tensor, x_count: torch.Tensor) -> torch.Tensor:
    """Count each state's rows of an owned row kind.

    Parameters
    ----------
    x_count_token : torch.Tensor
        (N,) int16. Each token's rows of the owned row kind.
    x_count : torch.Tensor
        (B,) long. How many tokens each state holds.

    Returns
    -------
    torch.Tensor
        (B,) long. Each state's rows of the owned row kind: its tokens' `x_count_token` summed.
    """
    # The rows owned by each state's tokens and every token before them
    x_rows_through = F.pad(x_count_token.cumsum(0), (1, 0))[x_count.cumsum(0)]  # (B,)
    return x_rows_through.diff(
        prepend=torch.zeros(1, dtype=torch.long, device=x_rows_through.device)
    )


def _taken(x_count: torch.Tensor, x_states: torch.Tensor, num_rows: int) -> _FieldFn:
    """Return the function gathering a block's field's rows of the states `x_states`.

    Parameters
    ----------
    x_count : torch.Tensor
        (B,) long. How many rows each of the block's states holds.
    x_states : torch.Tensor
        (S,) long. The states to take.
    num_rows : int
        The taken states' rows in all.

    Returns
    -------
    _FieldFn
        The function, gathering the states' rows state after state.
    """
    x_first_row = x_count.cumsum(0) - x_count
    x_rows = _rows_of_states(x_first_row[x_states], x_count[x_states], num_rows)
    return lambda x_field: x_field[x_rows]


def _joined(x_count: torch.Tensor, num_rows: int, num_regions: int) -> _FieldFn:
    """Return the function gathering, in NumPy, a shared step's block's field's leading rows.

    Parameters
    ----------
    x_count : torch.Tensor
        (B,) long. How many rows each of the block's states holds.
    num_rows : int
        The block's rows in all regions.
    num_regions : int
        The number of equal regions the block's rows split into.

    Returns
    -------
    _FieldFn
        The function gathering the leading rows of each region, region after region, as many as
        the region's states' `x_count`.
    """
    x_count_region = x_count.view(num_regions, -1).sum(1)  # (R,)
    x_first_row = torch.arange(num_regions) * (num_rows // num_regions)  # (R,)
    np_row = _rows_of_states(x_first_row, x_count_region, int(x_count_region.sum())).numpy()
    return lambda x_field: torch.from_numpy(x_field.numpy()[np_row])


def _copy_rows(x_region_field: torch.Tensor, x_field: torch.Tensor) -> None:
    """Copy `x_field` into the leading rows of its region's field.

    Where the region's rows hold more sparse entries, the ones past `x_field`'s are (0, 0.0)
    entries.

    Parameters
    ----------
    x_region_field : torch.Tensor
        The region's field, its leading rows written in place.
    x_field : torch.Tensor
        The field to copy, of the region's field's dtype.
    """
    assert x_region_field.dtype == x_field.dtype
    x_rows = x_region_field[: len(x_field)]
    if x_rows.shape == x_field.shape:
        x_rows.copy_(x_field)
    else:
        x_rows[:, : x_field.shape[1]] = x_field
        x_rows[:, x_field.shape[1] :] = 0


def _cut_entries(block: _Block) -> None:
    """Cut `block`'s entries to the most a row uses, at least one.

    A row's nonzero entries come first.

    Parameters
    ----------
    block : _Block
        The block, cut in place.
    """
    num_entries = max(int(block.x_value.numpy().any(axis=0).sum()), 1)
    block.x_column = block.x_column[:, :num_entries]
    block.x_value = block.x_value[:, :num_entries]


def _widened(x_gss: list[X_GameState]) -> list[X_GameState]:
    """Pad every block's entries to the widest of the states', as the encoders pad.

    Parameters
    ----------
    x_gss : list[X_GameState]
        The states.

    Returns
    -------
    list[X_GameState]
        The states, with blocks of their own, every block's entries padded to the widest of the
        states' with (0, 0.0) entries.
    """
    num_entries_block = [
        max(block.x_value.shape[1] for block in blocks) for blocks in zip(*map(_blocks, x_gss))
    ]
    xs = [_map_fields(x, lambda x_field: x_field) for x in x_gss]
    for x in xs:
        for block, num_entries in zip(_blocks(x), num_entries_block):
            block.x_column = F.pad(block.x_column, (0, num_entries - block.x_column.shape[1]))
            block.x_value = F.pad(block.x_value, (0, num_entries - block.x_value.shape[1]))
    return xs


def _owned(x: X_Tokens) -> list[X_OwnedSparse]:
    """Return the tokens' owned row kinds, in X_Tokens' order.

    Parameters
    ----------
    x : X_Tokens
        The tokens.

    Returns
    -------
    list[X_OwnedSparse]
        The owned row kinds: `effect_rows`, then `modifier_rows`.
    """
    return [x.effect_rows, x.modifier_rows]


def _blocks(x: X_GameState) -> list[_Block]:
    """Return every block of `x`: the tokens, their owned row kinds and the pending Effects.

    Parameters
    ----------
    x : X_GameState
        The game state.

    Returns
    -------
    list[_Block]
        The blocks, in that order.
    """
    return [x.tokens, *_owned(x.tokens), x.effect_pending]


def _fields(x: X_GameState) -> list[torch.Tensor]:
    """Return every field of `x`, in the order `_map_rows` maps them, without rebuilding a block.

    Parameters
    ----------
    x : X_GameState
        The game state.

    Returns
    -------
    list[torch.Tensor]
        The fields.
    """
    x_tokens, x_effect_pending = x.tokens, x.effect_pending
    return [
        x_tokens.x_column,
        x_tokens.x_value,
        x_tokens.x_token_kind,
        x_tokens.x_legal,
        x_tokens.x_needs_target,
        x_tokens.x_node_idx,
        *(
            x_field
            for x_owned in _owned(x_tokens)
            for x_field in (x_owned.x_column, x_owned.x_value, x_owned.x_count)
        ),
        x_tokens.x_count,
        x_effect_pending.x_column,
        x_effect_pending.x_value,
        x_effect_pending.x_count,
        x.energy,
        x.map_grid,
        x.map_meta,
        x.focus,
        x.ascension,
        x.reward_meta,
        x.shop_meta,
        x.event_meta,
        x.x_legal_action_type,
    ]


def _map_fields(x: X_GameState, fn: _FieldFn) -> X_GameState:
    """Apply `fn` to every field of `x`, in `_fields`' order.

    Parameters
    ----------
    x : X_GameState
        The game state.
    fn : _FieldFn
        The function on every field.

    Returns
    -------
    X_GameState
        `x` with every field through `fn`.
    """
    return _map_rows(x, fn, [fn for _ in _owned(x.tokens)], fn, fn)


def _map_rows(
    x: X_GameState,
    fn_token: _FieldFn,
    fns_owned: list[_FieldFn],
    fn_effect_pending: _FieldFn,
    fn_state: _FieldFn,
) -> X_GameState:
    """Map every field of `x` through the function of its rows, in `_fields`' order.

    Parameters
    ----------
    x : X_GameState
        The game state.
    fn_token : _FieldFn
        The function on the token rows' fields, the owned row kinds' counts among them.
    fns_owned : list[_FieldFn]
        Each owned row kind's function on its rows, in X_Tokens' order.
    fn_effect_pending : _FieldFn
        The function on the pending Effects' rows.
    fn_state : _FieldFn
        The function on the per-state fields, the tokens' and the pending Effects' counts among
        them.

    Returns
    -------
    X_GameState
        `x` with every field through the function of its rows.
    """
    x_tokens, x_effect_pending = x.tokens, x.effect_pending
    fn_effect_rows, fn_modifier_rows = fns_owned
    return X_GameState(
        tokens=X_Tokens(
            x_column=fn_token(x_tokens.x_column),
            x_value=fn_token(x_tokens.x_value),
            x_token_kind=fn_token(x_tokens.x_token_kind),
            x_legal=fn_token(x_tokens.x_legal),
            x_needs_target=fn_token(x_tokens.x_needs_target),
            x_node_idx=fn_token(x_tokens.x_node_idx),
            effect_rows=_map_owned(x_tokens.effect_rows, fn_effect_rows, fn_token),
            modifier_rows=_map_owned(x_tokens.modifier_rows, fn_modifier_rows, fn_token),
            x_count=fn_state(x_tokens.x_count),
        ),
        effect_pending=X_EffectPending(
            x_column=fn_effect_pending(x_effect_pending.x_column),
            x_value=fn_effect_pending(x_effect_pending.x_value),
            x_count=fn_state(x_effect_pending.x_count),
        ),
        energy=fn_state(x.energy),
        map_grid=fn_state(x.map_grid),
        map_meta=fn_state(x.map_meta),
        focus=fn_state(x.focus),
        ascension=fn_state(x.ascension),
        reward_meta=fn_state(x.reward_meta),
        shop_meta=fn_state(x.shop_meta),
        event_meta=fn_state(x.event_meta),
        x_legal_action_type=fn_state(x.x_legal_action_type),
    )


def _map_owned(x: X_OwnedSparse, fn_rows: _FieldFn, fn_token: _FieldFn) -> X_OwnedSparse:
    """Map an owned row kind's rows through `fn_rows` and its counts through `fn_token`.

    Parameters
    ----------
    x : X_OwnedSparse
        The owned row kind.
    fn_rows : _FieldFn
        The function on its rows.
    fn_token : _FieldFn
        The token rows' function, on its counts.

    Returns
    -------
    X_OwnedSparse
        The owned row kind with its rows through `fn_rows`, and its counts through the token
        rows' `fn_token`.
    """
    return X_OwnedSparse(
        x_column=fn_rows(x.x_column), x_value=fn_rows(x.x_value), x_count=fn_token(x.x_count)
    )


def _bytes_as(buffer: torch.Tensor, start: int, like: torch.Tensor) -> torch.Tensor:
    """View the bytes of `buffer` from `start` as a tensor of `like`'s dtype and shape.

    Parameters
    ----------
    buffer : torch.Tensor
        (number of bytes,) uint8. The bytes to view.
    start : int
        The view's first byte in `buffer`.
    like : torch.Tensor
        The tensor whose dtype and shape the view takes.

    Returns
    -------
    torch.Tensor
        The view of the bytes, in `like`'s shape and dtype.
    """
    n = like.numel() * like.element_size()
    return buffer[start : start + n].view(like.dtype).view(like.shape)
