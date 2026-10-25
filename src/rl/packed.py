"""The game state's rows: its states taken and joined, the shared step's regions written and
joined, and its one copy to a device"""

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
    """The states `x_states` (S,) of `x_gs`, in that order, packed on their own: the
    minibatches and their shuffle. `x_count_total` holds their rows in each block, in
    count_blocks' order, summed on the CPU from the counts there: the sizes, which would
    otherwise be read back"""
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
    """Each state's rows in each block, (B, number of blocks), the blocks in _blocks' order: the
    rollout keeps them on the CPU, where the minibatches sum take_states' sizes"""
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
    """The states one after another, as one: the rollout joins its steps"""
    xs = _widened(x_gss)
    x_fields = iter([torch.cat(x_fields_states) for x_fields_states in zip(*map(_fields, xs))])
    return _map_fields(xs[0], lambda x_field: next(x_fields))


def to_device(x_gs: X_GameState, device: torch.device) -> X_GameState:
    """`x_gs` copied onto `device` in one copy; on the CPU, the buffer is the copy"""

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
    """`x_gs` with every field moved to shared memory"""
    return _map_fields(x_gs, torch.Tensor.share_memory_)


def region_of_states(x_gs: X_GameState, first_state: int, num_states: int) -> X_GameState:
    """The region of the states [first_state, first_state + num_states) in a shared step's
    `x_gs`, whose every field holds the same rows per state: their counts and per-state
    fields, and their share of every block's rows, views all"""
    num_states_step = len(x_gs.tokens.x_count)

    def region(x_field: torch.Tensor) -> torch.Tensor:
        num_rows_state = len(x_field) // num_states_step
        return x_field[first_state * num_rows_state : (first_state + num_states) * num_rows_state]

    return _map_fields(x_gs, region)


def write_rows(x_region: X_GameState, x_gs: X_GameState) -> None:
    """`x_gs` written into a shared step's region, its states': every block's rows at the
    region's start, and the counts and per-state fields beside them"""
    for x_region_field, x_field in zip(_fields(x_region), _fields(x_gs)):
        _copy_rows(x_region_field, x_field)


def join_regions(x_gs: X_GameState, num_regions: int) -> X_GameState:
    """A shared step's `x_gs` of `num_regions` equal regions as one state: every block's
    leading rows of each region, region after region, as its counts say, its sparse entries cut
    to the most a row uses; the states' counts and the per-state fields as they are"""
    x = x_gs
    x_count = x.tokens.x_count
    fn_token = _joined(x_count, len(x.tokens.x_token_kind), num_regions)

    # An owned family's states hold the rows their joined tokens' counts say
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
    """The rows of states laid out one after another, `x_count` (S,) rows each, each state's from
    its `x_first_row` (S,) on; `num_rows`, their total, sizes the output without a read back"""
    x_shift = x_first_row - (x_count.cumsum(0) - x_count)
    x_shift_row = x_shift.repeat_interleave(x_count, output_size=num_rows)
    return torch.arange(num_rows, device=x_count.device) + x_shift_row


def _count_state(x_count_token: torch.Tensor, x_count: torch.Tensor) -> torch.Tensor:
    """Each state's rows of an owned family, (B,): its tokens' `x_count_token` (N,) summed, the
    states holding `x_count` (B,) tokens each"""
    # The rows owned by each state's tokens and every token before them
    x_rows_through = F.pad(x_count_token.cumsum(0), (1, 0))[x_count.cumsum(0)]  # (B,)
    return x_rows_through.diff(prepend=x_rows_through.new_zeros(1))


def _taken(x_count: torch.Tensor, x_states: torch.Tensor, num_rows: int) -> _FieldFn:
    """The function gathering a block's field's rows of the states `x_states`, state after state:
    the block's states hold `x_count` rows each, the taken ones `num_rows` in all"""
    x_first_row = x_count.cumsum(0) - x_count
    x_rows = _rows_of_states(x_first_row[x_states], x_count[x_states], num_rows)
    return lambda x_field: x_field[x_rows]


def _joined(x_count: torch.Tensor, num_rows: int, num_regions: int) -> _FieldFn:
    """The function gathering, in NumPy, a shared step's block's field's leading rows of each of
    the `num_regions` equal regions of its `num_rows` rows, region after region, as many as the
    region's states' `x_count`"""
    x_count_region = x_count.view(num_regions, -1).sum(1)  # (R,)
    x_first_row = torch.arange(num_regions) * (num_rows // num_regions)  # (R,)
    np_row = _rows_of_states(x_first_row, x_count_region, int(x_count_region.sum())).numpy()
    return lambda x_field: torch.from_numpy(x_field.numpy()[np_row])


def _copy_rows(x_region_field: torch.Tensor, x_field: torch.Tensor) -> None:
    """Copy `x_field` into the leading rows of its region's field; where the region's rows hold
    more sparse entries, the ones past `x_field`'s are (0, 0.0) entries"""
    assert x_region_field.dtype == x_field.dtype
    x_rows = x_region_field[: len(x_field)]
    if x_rows.shape == x_field.shape:
        x_rows.copy_(x_field)
    else:
        x_rows[:, : x_field.shape[1]] = x_field
        x_rows[:, x_field.shape[1] :] = 0


def _cut_entries(block: _Block) -> None:
    """`block`'s entries cut to the most a row uses, at least one: a row's nonzero entries come
    first"""
    num_entries = max(int(block.x_value.numpy().any(axis=0).sum()), 1)
    block.x_column = block.x_column[:, :num_entries]
    block.x_value = block.x_value[:, :num_entries]


def _widened(x_gss: list[X_GameState]) -> list[X_GameState]:
    """The states, with blocks of their own, every block's entries padded to the widest of the
    states' with (0, 0.0) entries, as the encoders pad"""
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
    """The tokens' owned families, in X_Tokens' order"""
    return [x.effects, x.modifiers]


def _blocks(x: X_GameState) -> list[_Block]:
    """Every block of `x`: the tokens, their owned families and the pending effects"""
    return [x.tokens, *_owned(x.tokens), x.effect_pending]


def _fields(x: X_GameState) -> list[torch.Tensor]:
    """Every field of `x`, in the order _map_rows maps them, without rebuilding a block"""
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
    """`fn` on every field of `x`, in _fields' order"""
    return _map_rows(x, fn, [fn for _ in _owned(x.tokens)], fn, fn)


def _map_rows(
    x: X_GameState,
    fn_token: _FieldFn,
    fns_owned: list[_FieldFn],
    fn_effect_pending: _FieldFn,
    fn_state: _FieldFn,
) -> X_GameState:
    """`x` with every field through the function of its rows, in _fields' order: `fn_token` on
    the token rows' fields, the owned families' counts among them; `fns_owned` on each owned
    family's rows, in X_Tokens' order; `fn_effect_pending` on the pending effects' rows; and
    `fn_state` on the per-state fields, the tokens' and the pending effects' counts among them"""
    x_tokens, x_effect_pending = x.tokens, x.effect_pending
    fn_effects, fn_modifiers = fns_owned
    return X_GameState(
        tokens=X_Tokens(
            x_column=fn_token(x_tokens.x_column),
            x_value=fn_token(x_tokens.x_value),
            x_token_kind=fn_token(x_tokens.x_token_kind),
            x_legal=fn_token(x_tokens.x_legal),
            x_needs_target=fn_token(x_tokens.x_needs_target),
            x_node_idx=fn_token(x_tokens.x_node_idx),
            effects=_map_owned(x_tokens.effects, fn_effects, fn_token),
            modifiers=_map_owned(x_tokens.modifiers, fn_modifiers, fn_token),
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
    """An owned family with its rows through `fn_rows`, and its counts through the token rows'
    `fn_token`"""
    return X_OwnedSparse(
        x_column=fn_rows(x.x_column), x_value=fn_rows(x.x_value), x_count=fn_token(x.x_count)
    )


def _bytes_as(buffer: torch.Tensor, start: int, like: torch.Tensor) -> torch.Tensor:
    """The bytes of `buffer` from `start` viewed as a tensor of `like`'s dtype and shape"""
    n = like.numel() * like.element_size()
    return buffer[start : start + n].view(like.dtype).view(like.shape)
