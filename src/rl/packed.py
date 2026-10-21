"""The game state's rows: its states taken and joined, the shared step's regions written and
joined, and its one copy to a device"""

import copy
from collections.abc import Callable
from typing import TypeVar

import torch
import torch.nn.functional as F

from src.rl.types import X_EffectPending
from src.rl.types import X_GameState
from src.rl.types import X_OwnedSparse
from src.rl.types import X_Tokens


# A block of the state, its rows held sparse
_Block = TypeVar("_Block", X_Tokens, X_OwnedSparse, X_EffectPending)


def take_states(
    x_gs: X_GameState, x_states: torch.Tensor, x_count_total: torch.Tensor
) -> X_GameState:
    """The states `x_states` (S,) of `x_gs`, in that order, packed on their own: the
    minibatches and their shuffle. `x_count_total` holds their rows in each block, in
    count_blocks' order, summed on the CPU from the counts there: the sizes, which would
    otherwise be read back"""
    (
        num_tokens,
        num_health_block,
        num_prices,
        num_effects,
        num_modifiers,
        num_effect_pending,
    ) = x_count_total.tolist()
    x = x_gs
    return X_GameState(
        tokens=_take_rows(x.tokens, x_states, num_tokens),
        health_block=_take_rows(x.health_block, x_states, num_health_block),
        prices=_take_rows(x.prices, x_states, num_prices),
        effects=_take_rows(x.effects, x_states, num_effects),
        modifiers=_take_rows(x.modifiers, x_states, num_modifiers),
        effect_pending=_take_rows(x.effect_pending, x_states, num_effect_pending),
        energy=x.energy[x_states],
        map_grid=x.map_grid[x_states],
        map_meta=x.map_meta[x_states],
        focus=x.focus[x_states],
        ascension=x.ascension[x_states],
        reward_meta=x.reward_meta[x_states],
        shop_meta=x.shop_meta[x_states],
        event_meta=x.event_meta[x_states],
        x_legal_action_type=x.x_legal_action_type[x_states],
    )


def count_blocks(x_gs: X_GameState) -> torch.Tensor:
    """Each state's rows in each block, (B, number of blocks), the blocks in X_GameState's order:
    the rollout keeps them on the CPU, where the minibatches sum take_states' sizes"""
    x = x_gs
    return torch.stack(
        [
            x.tokens.x_count,
            x.health_block.x_count,
            x.prices.x_count,
            x.effects.x_count,
            x.modifiers.x_count,
            x.effect_pending.x_count,
        ],
        dim=1,
    )


def cat_states(x_gss: list[X_GameState]) -> X_GameState:
    """The states one after another, as one: the rollout joins its steps"""
    xs = x_gss
    return X_GameState(
        tokens=_cat_rows(_widened([x.tokens for x in xs])),
        health_block=_cat_rows(_widened([x.health_block for x in xs])),
        prices=_cat_rows(_widened([x.prices for x in xs])),
        effects=_cat_rows(_widened([x.effects for x in xs])),
        modifiers=_cat_rows(_widened([x.modifiers for x in xs])),
        effect_pending=_cat_rows(_widened([x.effect_pending for x in xs])),
        energy=torch.cat([x.energy for x in xs]),
        map_grid=torch.cat([x.map_grid for x in xs]),
        map_meta=torch.cat([x.map_meta for x in xs]),
        focus=torch.cat([x.focus for x in xs]),
        ascension=torch.cat([x.ascension for x in xs]),
        reward_meta=torch.cat([x.reward_meta for x in xs]),
        shop_meta=torch.cat([x.shop_meta for x in xs]),
        event_meta=torch.cat([x.event_meta for x in xs]),
        x_legal_action_type=torch.cat([x.x_legal_action_type for x in xs]),
    )


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
    leading rows of each region, region after region, its sparse entries cut to the most a row
    uses, as its count says; the counts and the per-state fields as they are"""
    x = x_gs
    return X_GameState(
        tokens=_cut_entries(_join_rows(x.tokens, num_regions)),
        health_block=_cut_entries(_join_rows(x.health_block, num_regions)),
        prices=_cut_entries(_join_rows(x.prices, num_regions)),
        effects=_cut_entries(_join_rows(x.effects, num_regions)),
        modifiers=_cut_entries(_join_rows(x.modifiers, num_regions)),
        effect_pending=_cut_entries(_join_rows(x.effect_pending, num_regions)),
        energy=x.energy,
        map_grid=x.map_grid,
        map_meta=x.map_meta,
        focus=x.focus,
        ascension=x.ascension,
        reward_meta=x.reward_meta,
        shop_meta=x.shop_meta,
        event_meta=x.event_meta,
        x_legal_action_type=x.x_legal_action_type,
    )


def _rows_of_states(
    x_first_row: torch.Tensor, x_count: torch.Tensor, num_rows: int
) -> torch.Tensor:
    """The rows of states laid out one after another, `x_count` (S,) rows each, each state's from
    its `x_first_row` (S,) on; `num_rows`, their total, sizes the output without a read back"""
    x_shift = x_first_row - (x_count.cumsum(0) - x_count)
    x_shift_row = x_shift.repeat_interleave(x_count, output_size=num_rows)
    return torch.arange(num_rows, device=x_count.device) + x_shift_row


def _join_rows(block: _Block, num_regions: int) -> _Block:
    """`block`'s leading rows of each of its `num_regions` equal regions, region after region, as
    many as the region's states' `x_count`: gathered in NumPy"""
    x_row_fields = _row_fields(block)
    num_rows_region = len(x_row_fields[0]) // num_regions  # every row field holds all the rows
    x_count_region = block.x_count.view(num_regions, -1).sum(1)  # (R,)
    x_first_row = torch.arange(num_regions) * num_rows_region  # (R,)
    np_row = _rows_of_states(x_first_row, x_count_region, int(x_count_region.sum())).numpy()
    return _block_from_row_fields(
        block,
        [torch.from_numpy(x_field.numpy()[np_row]) for x_field in x_row_fields],
        block.x_count,
    )


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


def _take_rows(block: _Block, x_states: torch.Tensor, num_rows: int) -> _Block:
    """`block`'s rows of the states `x_states`, state after state, and their counts: its states
    hold `x_count` rows each, the taken ones `num_rows` in all"""
    x_count = block.x_count
    x_count_taken = x_count[x_states]
    x_first_row = x_count.cumsum(0) - x_count
    x_rows = _rows_of_states(x_first_row[x_states], x_count_taken, num_rows)
    return _block_from_row_fields(
        block, [x_field[x_rows] for x_field in _row_fields(block)], x_count_taken
    )


def _cut_entries(rows: _Block) -> _Block:
    """`rows` with their entries cut to the most a row uses, at least one: a row's nonzero entries
    come first"""
    num_entries = max(int(rows.x_value.numpy().any(axis=0).sum()), 1)
    rows.x_column = rows.x_column[:, :num_entries]
    rows.x_value = rows.x_value[:, :num_entries]
    return rows


def _cat_rows(blocks: list[_Block]) -> _Block:
    """The blocks' rows one after another, and their counts"""
    return _block_from_row_fields(
        blocks[0],
        [torch.cat(x_fields) for x_fields in zip(*[_row_fields(block) for block in blocks])],
        torch.cat([block.x_count for block in blocks]),
    )


def _widened(blocks: list[_Block]) -> list[_Block]:
    """The blocks, every block's entries padded to the widest with (0, 0.0) entries, as the
    encoders pad"""
    num_entries = max(block.x_value.shape[1] for block in blocks)
    widened = []
    for block in blocks:
        block = copy.copy(block)
        block.x_column = F.pad(block.x_column, (0, num_entries - block.x_column.shape[1]))
        block.x_value = F.pad(block.x_value, (0, num_entries - block.x_value.shape[1]))
        widened.append(block)
    return widened


def _fields(x: X_GameState) -> list[torch.Tensor]:
    """Every field of `x`, block by block, in _map_fields' order, without rebuilding a block"""
    return [
        *_block_fields(x.tokens),
        *_block_fields(x.health_block),
        *_block_fields(x.prices),
        *_block_fields(x.effects),
        *_block_fields(x.modifiers),
        *_block_fields(x.effect_pending),
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


def _map_fields(x: X_GameState, fn: Callable[[torch.Tensor], torch.Tensor]) -> X_GameState:
    """`fn` on every field of `x`, block by block, in a fixed order"""
    return X_GameState(
        tokens=_map_block(x.tokens, fn),
        health_block=_map_block(x.health_block, fn),
        prices=_map_block(x.prices, fn),
        effects=_map_block(x.effects, fn),
        modifiers=_map_block(x.modifiers, fn),
        effect_pending=_map_block(x.effect_pending, fn),
        energy=fn(x.energy),
        map_grid=fn(x.map_grid),
        map_meta=fn(x.map_meta),
        focus=fn(x.focus),
        ascension=fn(x.ascension),
        reward_meta=fn(x.reward_meta),
        shop_meta=fn(x.shop_meta),
        event_meta=fn(x.event_meta),
        x_legal_action_type=fn(x.x_legal_action_type),
    )


def _block_fields(block: _Block) -> list[torch.Tensor]:
    """Every field of `block`, in _map_block's order: its row fields, then its count"""
    return [*_row_fields(block), block.x_count]


def _map_block(block: _Block, fn: Callable[[torch.Tensor], torch.Tensor]) -> _Block:
    """`fn` on every row field of `block`, then on its count"""
    return _block_from_row_fields(
        block, [fn(x_field) for x_field in _row_fields(block)], fn(block.x_count)
    )


def _row_fields(block: _Block) -> list[torch.Tensor]:
    """`block`'s row fields, every field but its count, in its block type's fixed order"""
    match block:
        case X_Tokens():
            return _row_fields_tokens(block)
        case X_OwnedSparse():
            return _row_fields_owned_sparse(block)
        case X_EffectPending():
            return _row_fields_effect_pending(block)


def _block_from_row_fields(
    like: _Block, x_row_fields: list[torch.Tensor], x_count: torch.Tensor
) -> _Block:
    """A block of `like`'s type from its row fields, in _row_fields' order, and its count"""
    match like:
        case X_Tokens():
            return _tokens_from_row_fields(x_row_fields, x_count)
        case X_OwnedSparse():
            return _owned_sparse_from_row_fields(x_row_fields, x_count)
        case X_EffectPending():
            return _effect_pending_from_row_fields(x_row_fields, x_count)


def _row_fields_tokens(x: X_Tokens) -> list[torch.Tensor]:
    return [x.x_column, x.x_value, x.x_token_kind, x.x_legal, x.x_needs_target, x.x_node_idx]


def _tokens_from_row_fields(x_row_fields: list[torch.Tensor], x_count: torch.Tensor) -> X_Tokens:
    x_column, x_value, x_token_kind, x_legal, x_needs_target, x_node_idx = x_row_fields
    return X_Tokens(
        x_column=x_column,
        x_value=x_value,
        x_token_kind=x_token_kind,
        x_legal=x_legal,
        x_needs_target=x_needs_target,
        x_node_idx=x_node_idx,
        x_count=x_count,
    )


def _row_fields_owned_sparse(x: X_OwnedSparse) -> list[torch.Tensor]:
    return [x.x_column, x.x_value, x.x_owner_offset]


def _owned_sparse_from_row_fields(
    x_row_fields: list[torch.Tensor], x_count: torch.Tensor
) -> X_OwnedSparse:
    x_column, x_value, x_owner_offset = x_row_fields
    return X_OwnedSparse(
        x_column=x_column, x_value=x_value, x_owner_offset=x_owner_offset, x_count=x_count
    )


def _row_fields_effect_pending(x: X_EffectPending) -> list[torch.Tensor]:
    return [x.x_column, x.x_value]


def _effect_pending_from_row_fields(
    x_row_fields: list[torch.Tensor], x_count: torch.Tensor
) -> X_EffectPending:
    x_column, x_value = x_row_fields
    return X_EffectPending(x_column=x_column, x_value=x_value, x_count=x_count)


def _bytes_as(buffer: torch.Tensor, start: int, like: torch.Tensor) -> torch.Tensor:
    """The bytes of `buffer` from `start` viewed as a tensor of `like`'s dtype and shape"""
    n = like.numel() * like.element_size()
    return buffer[start : start + n].view(like.dtype).view(like.shape)
