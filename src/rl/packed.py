"""The game state's rows: its states taken and joined, the shared step's regions written and
joined, and its one copy to a device"""

from collections.abc import Callable
from typing import TypeVar

import torch
import torch.nn.functional as F

from src.rl.types import X_Count
from src.rl.types import X_EffectPending
from src.rl.types import X_EntitySparse
from src.rl.types import X_GameState
from src.rl.types import X_OwnedSparse
from src.rl.types import X_Rooms


# A token-like block of the state, and one held sparse
Block = TypeVar("Block", X_EntitySparse, X_Rooms, X_OwnedSparse, X_EffectPending)
BlockSparse = TypeVar("BlockSparse", X_EntitySparse, X_OwnedSparse, X_EffectPending)

# A field of the shared step's state: a block, the counts or a per-state field
Field = TypeVar(
    "Field", X_EntitySparse, X_Rooms, X_OwnedSparse, X_EffectPending, X_Count, torch.Tensor
)


def take_states(
    x_game_state: X_GameState, x_states: torch.Tensor, x_count_total: X_Count
) -> X_GameState:
    """The states `x_states` (S,) of `x_game_state`, in that order, packed on their own: the
    minibatches and their shuffle. `x_count_total` holds their rows in each block, summed on the
    CPU from the counts there: the sizes, which would otherwise be read back"""
    x = x_game_state
    return X_GameState(
        cards=_take_rows(x.cards, x.count.cards, x_states, int(x_count_total.cards)),
        relics=_take_rows(x.relics, x.count.relics, x_states, int(x_count_total.relics)),
        potions=_take_rows(x.potions, x.count.potions, x_states, int(x_count_total.potions)),
        monsters=_take_rows(x.monsters, x.count.monsters, x_states, int(x_count_total.monsters)),
        event_options=_take_rows(
            x.event_options, x.count.event_options, x_states, int(x_count_total.event_options)
        ),
        character=_take_rows(
            x.character, x.count.character, x_states, int(x_count_total.character)
        ),
        rooms=_take_rows(x.rooms, x.count.rooms, x_states, int(x_count_total.rooms)),
        health_block=_take_rows(
            x.health_block, x.count.health_block, x_states, int(x_count_total.health_block)
        ),
        prices=_take_rows(x.prices, x.count.prices, x_states, int(x_count_total.prices)),
        effects=_take_rows(x.effects, x.count.effects, x_states, int(x_count_total.effects)),
        effect_pending=_take_rows(
            x.effect_pending, x.count.effect_pending, x_states, int(x_count_total.effect_pending)
        ),
        modifiers=_take_rows(
            x.modifiers, x.count.modifiers, x_states, int(x_count_total.modifiers)
        ),
        count=x.count[x_states],
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


def count_total(x_count: X_Count) -> X_Count:
    """The rows of `x_count`'s states in each block, in all: take_states' sizes"""
    return x_count.apply(lambda x_count_block: x_count_block.sum(), batch_size=[])


def cat_states(x_game_states: list[X_GameState]) -> X_GameState:
    """The states one after another, as one: the rollout joins its steps"""
    xs = x_game_states
    return X_GameState(
        cards=_cat_rows([x.cards for x in xs]),
        relics=_cat_rows([x.relics for x in xs]),
        potions=_cat_rows([x.potions for x in xs]),
        monsters=_cat_rows([x.monsters for x in xs]),
        event_options=_cat_rows([x.event_options for x in xs]),
        character=_cat_rows([x.character for x in xs]),
        rooms=torch.cat([x.rooms for x in xs]),
        health_block=_cat_rows([x.health_block for x in xs]),
        prices=_cat_rows([x.prices for x in xs]),
        effects=_cat_rows([x.effects for x in xs]),
        effect_pending=_cat_rows([x.effect_pending for x in xs]),
        modifiers=_cat_rows([x.modifiers for x in xs]),
        count=torch.cat([x.count for x in xs]),
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


def to_device(x_game_state: X_GameState, device: torch.device) -> X_GameState:
    """`x_game_state` copied onto `device` in one copy; on the CPU, the buffer is the copy"""

    # Every field packed into one pinned buffer and sent in one copy: a copy per field from
    # pageable memory makes the CPU wait for the GPU once per field
    fields = _fields(x_game_state)
    starts, end = [], 0
    for t in fields:
        starts.append(end)
        end += -(-t.numel() * t.element_size() // 8) * 8  # 8-byte aligned for every dtype
    staging = torch.empty(end, dtype=torch.uint8, pin_memory=device.type == "cuda")
    for t, start in zip(fields, starts):
        _bytes_as(staging, start, t).copy_(t)
    packed = staging.to(device, non_blocking=True) if device.type == "cuda" else staging
    pieces = iter(_bytes_as(packed, start, t) for t, start in zip(fields, starts))
    return _map_fields(x_game_state, lambda x_field: next(pieces))


def share_memory(x_game_state: X_GameState) -> X_GameState:
    """`x_game_state` with every field moved to shared memory"""
    return _map_fields(x_game_state, torch.Tensor.share_memory_)


def region_of_states(x_game_state: X_GameState, first_state: int, num_states: int) -> X_GameState:
    """The region of the states [first_state, first_state + num_states) in a shared step's
    `x_game_state`, whose every field holds the same rows per state: their counts and per-state
    fields, and their share of every block's rows, views all"""
    num_states_step = len(x_game_state.count)

    def region(x_field: Field) -> Field:
        num_rows_state = len(x_field) // num_states_step
        return x_field[first_state * num_rows_state : (first_state + num_states) * num_rows_state]

    x = x_game_state
    return X_GameState(
        cards=region(x.cards),
        relics=region(x.relics),
        potions=region(x.potions),
        monsters=region(x.monsters),
        event_options=region(x.event_options),
        character=region(x.character),
        rooms=region(x.rooms),
        health_block=region(x.health_block),
        prices=region(x.prices),
        effects=region(x.effects),
        effect_pending=region(x.effect_pending),
        modifiers=region(x.modifiers),
        count=region(x.count),
        energy=region(x.energy),
        map_grid=region(x.map_grid),
        map_meta=region(x.map_meta),
        focus=region(x.focus),
        ascension=region(x.ascension),
        reward_meta=region(x.reward_meta),
        shop_meta=region(x.shop_meta),
        event_meta=region(x.event_meta),
        x_legal_action_type=region(x.x_legal_action_type),
    )


def write_rows(x_region: X_GameState, x_game_state: X_GameState) -> None:
    """`x_game_state` written into a shared step's region, its states': every block's rows at the
    region's start, and the counts and per-state fields beside them"""
    for x_region_field, x_field in zip(_fields(x_region), _fields(x_game_state)):
        _copy_rows(x_region_field, x_field)


def join_regions(x_game_state: X_GameState, num_regions: int) -> X_GameState:
    """A shared step's `x_game_state` of `num_regions` equal regions as one state: every block's
    leading rows of each region, region after region, its sparse entries cut to the most a row
    uses, as `count` says; the counts copied, the per-state fields as they are"""
    x = x_game_state
    return X_GameState(
        cards=_cut_entries(_join_rows(x.cards, x.count.cards, num_regions)),
        relics=_cut_entries(_join_rows(x.relics, x.count.relics, num_regions)),
        potions=_cut_entries(_join_rows(x.potions, x.count.potions, num_regions)),
        monsters=_cut_entries(_join_rows(x.monsters, x.count.monsters, num_regions)),
        event_options=_cut_entries(
            _join_rows(x.event_options, x.count.event_options, num_regions)
        ),
        character=_cut_entries(_join_rows(x.character, x.count.character, num_regions)),
        rooms=_join_rows(x.rooms, x.count.rooms, num_regions),
        health_block=_cut_entries(_join_rows(x.health_block, x.count.health_block, num_regions)),
        prices=_cut_entries(_join_rows(x.prices, x.count.prices, num_regions)),
        effects=_cut_entries(_join_rows(x.effects, x.count.effects, num_regions)),
        effect_pending=_cut_entries(
            _join_rows(x.effect_pending, x.count.effect_pending, num_regions)
        ),
        modifiers=_cut_entries(_join_rows(x.modifiers, x.count.modifiers, num_regions)),
        count=x.count.clone(),
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


def rows_of_states(
    x_first_row: torch.Tensor, x_count: torch.Tensor, num_rows: int
) -> torch.Tensor:
    """The rows of states laid out one after another, `x_count` (S,) rows each, each state's from
    its `x_first_row` (S,) on; `num_rows`, their total, sizes the output without a read back"""
    x_shift = x_first_row - (x_count.cumsum(0) - x_count)
    x_shift_row = x_shift.repeat_interleave(x_count, output_size=num_rows)
    return torch.arange(num_rows, device=x_count.device) + x_shift_row


def _join_rows(block: Block, x_count: torch.Tensor, num_regions: int) -> Block:
    """`block`'s leading rows of each of its `num_regions` equal regions, region after region, as
    many as the region's states' `x_count`: gathered in NumPy"""
    x_count_region = x_count.view(num_regions, -1).sum(1)  # (R,)
    x_first_row = torch.arange(num_regions) * (len(block) // num_regions)  # (R,)
    np_row = rows_of_states(x_first_row, x_count_region, int(x_count_region.sum())).numpy()
    return block.apply(
        lambda x_field: torch.from_numpy(x_field.numpy()[np_row]), batch_size=[len(np_row)]
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


def _take_rows(
    block: Block, x_count: torch.Tensor, x_states: torch.Tensor, num_rows: int
) -> Block:
    """`block`'s rows of the states `x_states`, state after state: its states hold `x_count` rows
    each, the taken ones `num_rows` in all"""
    x_first_row = x_count.cumsum(0) - x_count
    return block[rows_of_states(x_first_row[x_states], x_count[x_states], num_rows)]


def _cut_entries(rows: BlockSparse) -> BlockSparse:
    """`rows` with their entries cut to the most a row uses, at least one: a row's nonzero entries
    come first"""
    num_entries = max(int(rows.x_value.numpy().any(axis=0).sum()), 1)
    rows.x_column = rows.x_column[:, :num_entries]
    rows.x_value = rows.x_value[:, :num_entries]
    return rows


def _cat_rows(blocks: list[BlockSparse]) -> BlockSparse:
    """The blocks' rows one after another, every block's entries padded to the widest with
    (0, 0.0) entries, as the encoders pad"""
    num_entries = max(block.x_value.shape[1] for block in blocks)
    widened = []
    for block in blocks:
        block = block.copy()
        block.x_column = F.pad(block.x_column, (0, num_entries - block.x_column.shape[1]))
        block.x_value = F.pad(block.x_value, (0, num_entries - block.x_value.shape[1]))
        widened.append(block)
    return torch.cat(widened)


def _fields(x: X_GameState) -> list[torch.Tensor]:
    """Every field of `x`, block by block, in _map_fields' order, without rebuilding a block"""
    return [
        *x.cards.values(),
        *x.relics.values(),
        *x.potions.values(),
        *x.monsters.values(),
        *x.event_options.values(),
        *x.character.values(),
        *x.rooms.values(),
        *x.health_block.values(),
        *x.prices.values(),
        *x.effects.values(),
        *x.effect_pending.values(),
        *x.modifiers.values(),
        *x.count.values(),
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
        cards=x.cards.apply(fn),
        relics=x.relics.apply(fn),
        potions=x.potions.apply(fn),
        monsters=x.monsters.apply(fn),
        event_options=x.event_options.apply(fn),
        character=x.character.apply(fn),
        rooms=x.rooms.apply(fn),
        health_block=x.health_block.apply(fn),
        prices=x.prices.apply(fn),
        effects=x.effects.apply(fn),
        effect_pending=x.effect_pending.apply(fn),
        modifiers=x.modifiers.apply(fn),
        count=x.count.apply(fn),
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


def _bytes_as(buffer: torch.Tensor, start: int, like: torch.Tensor) -> torch.Tensor:
    """The bytes of `buffer` from `start` viewed as a tensor of `like`'s dtype and shape"""
    n = like.numel() * like.element_size()
    return buffer[start : start + n].view(like.dtype).view(like.shape)
