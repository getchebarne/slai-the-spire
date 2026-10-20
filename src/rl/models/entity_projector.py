from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.encoding.card import ENCODING_DIM_CARD
from src.rl.encoding.character import ENCODING_DIM_CHARACTER
from src.rl.encoding.effect import ENCODING_DIM_EFFECT
from src.rl.encoding.event import ENCODING_DIM_EVENT_OPTION
from src.rl.encoding.focus import OFF_FOCUS_DRAW_ORDER_VISIBLE
from src.rl.encoding.health_block import ENCODING_DIM_HEALTH_BLOCK
from src.rl.encoding.modifier import ENCODING_DIM_MODIFIER
from src.rl.encoding.monster import ENCODING_DIM_MONSTER
from src.rl.encoding.potion import ENCODING_DIM_POTION
from src.rl.encoding.relic import ENCODING_DIM_RELIC
from src.rl.encoding.shop import ENCODING_DIM_PRICE
from src.rl.types import SliceKind
from src.rl.types import TokenCounts
from src.rl.types import X_DrawOrderStates
from src.rl.types import X_Entity
from src.rl.types import X_EntitySparse
from src.rl.types import X_GameState
from src.rl.types import X_OwnedSparse
from src.rl.types import X_Rooms
from src.rl.types import X_Window


class EntityProjector(nn.Module):
    """Factored (AlphaStar-style) entity embedder: a private first layer per block, the shared
    row families (health/block, prices, effects, modifiers) added into their owners, then one
    shared second layer + LayerNorm. Returns the window packed: its tokens (the six blocks then
    the rooms, state after state) and their X_Window; then the window's slots per state, its
    valid tokens per entity type, the draw-order layer's states, and the embedded pending halt."""

    def __init__(self, dim_entity: int, dim_map: int) -> None:
        super().__init__()

        self._dim_entity = dim_entity

        # Per-Entity kind private mappers
        self._pre_card = nn.Linear(ENCODING_DIM_CARD, dim_entity)
        self._pre_relic = nn.Linear(ENCODING_DIM_RELIC, dim_entity)
        self._pre_potion = nn.Linear(ENCODING_DIM_POTION, dim_entity)
        self._pre_event_option = nn.Linear(ENCODING_DIM_EVENT_OPTION, dim_entity)
        self._pre_monster = nn.Linear(ENCODING_DIM_MONSTER, dim_entity)
        self._pre_character = nn.Linear(ENCODING_DIM_CHARACTER, dim_entity)
        self._pre_room = nn.Linear(dim_map, dim_entity)  # Rooms come processed by `MapGNN`

        # Health and Block's mapper is shared between the Character and the Monsters
        self._pre_health_block = nn.Linear(ENCODING_DIM_HEALTH_BLOCK, dim_entity, bias=False)

        # Prices' mapper is shared between the Cards, Relics and Potions
        self._pre_price = nn.Linear(ENCODING_DIM_PRICE, dim_entity, bias=False)

        # Per-Effect embedder w/ >=2 layers is a must due to OHE-summing semantics
        self._effects_mlp = nn.Sequential(
            nn.Linear(ENCODING_DIM_EFFECT, 2 * dim_entity),
            nn.ReLU(),
            nn.Linear(2 * dim_entity, dim_entity, bias=False),
        )

        # Per-Modifier embedder, same shape / logic as the Effects'
        self._modifiers_mlp = nn.Sequential(
            nn.Linear(ENCODING_DIM_MODIFIER, 2 * dim_entity),
            nn.ReLU(),
            nn.Linear(2 * dim_entity, dim_entity, bias=False),
        )

        # Shared second layer (after ReLU, on the fused all-blocks row tensor)
        self._post_shared = nn.Linear(dim_entity, dim_entity)

        # LayerNorm cuz LayerNorm
        self._layer_norm = nn.LayerNorm(dim_entity)

    def forward(
        self, x_game_state: X_GameState, x_rooms: X_Entity
    ) -> tuple[torch.Tensor, X_Window, int, TokenCounts, X_DrawOrderStates, torch.Tensor]:
        # Shapes: B = batch, D = embedding width, T = the window's tokens, N = a family's
        # valid rows across the batch, N_tok = the window's valid tokens, P = the states with a
        # pending halt, A = the action types
        entity_blocks: list[X_EntitySparse | X_Rooms] = [
            x_game_state.cards,
            x_game_state.relics,
            x_game_state.potions,
            x_game_state.monsters,
            x_game_state.event_options,
            x_game_state.character,
            x_game_state.rooms,
        ]

        # X_GameState order: the list above IS the window's entity order, the order
        # every consumer cats the slice kind rows in
        x_slice_kind = torch.cat(
            [x_entity.x_slice_kind for x_entity in entity_blocks], dim=1
        ).long()  # (B, T)

        # The window's masks, token for token with its slice kind row
        x_legal = torch.cat([x_entity.x_legal for x_entity in entity_blocks], dim=1)  # (B, T, A)
        x_needs_target = torch.cat(
            [x_entity.x_needs_target for x_entity in entity_blocks], dim=1
        )  # (B, T)
        x_is_token = x_slice_kind >= 0  # (B, T)

        # Each state's tokens, and its tokens of each slice kind: the pads count in an extra column
        x_count = x_is_token.sum(dim=1)  # (B,)
        x_slice_kind_column = x_slice_kind.masked_fill(~x_is_token, len(SliceKind))  # (B, T)
        x_slice_kind_count = x_count.new_zeros(len(x_count), len(SliceKind) + 1)
        x_slice_kind_count.scatter_add_(1, x_slice_kind_column, torch.ones_like(x_slice_kind))
        x_slice_kind_count = x_slice_kind_count[:, :-1]  # (B, len(SliceKind))

        # The draw-order layer's states, whose draw pile shows an order (2 or more draw tokens):
        # `x_ordered` masks those under Frozen Eye (visible), then the others (hidden)
        x_count_pile = x_slice_kind_count[:, SliceKind.CARD_PILE_DRAW]  # (B,)
        x_draw_order_visible = x_game_state.focus[:, OFF_FOCUS_DRAW_ORDER_VISIBLE] > 0
        x_ordered = (x_count_pile >= 2) & torch.stack(
            [x_draw_order_visible, ~x_draw_order_visible]
        )  # (2, B)

        # Every pack's size in one read: the CPU waits for the GPU here, and nowhere else below
        # The owned-row families, in the order they add into their owners
        x_owned: list[X_OwnedSparse] = [
            x_game_state.prices,
            x_game_state.health_block,
            x_game_state.effects,
            x_game_state.modifiers,
        ]
        (
            num_cards,
            num_relics,
            num_potions,
            num_monsters,
            num_event_options,
            num_character,
            num_rooms,
            num_prices,
            num_health_block,
            num_effects,
            num_modifiers,
            num_pending,
            num_states_visible,
            num_states_hidden,
            num_rows_visible,
            num_rows_hidden,
            num_pile_visible,
            num_pile_hidden,
        ) = torch.cat(
            [
                torch.stack(
                    [x_entity.mask.sum() for x_entity in entity_blocks]
                    + [x_owned_sparse.mask.sum() for x_owned_sparse in x_owned]
                    + [x_game_state.effect_pending.mask.sum()]
                ),
                x_ordered.sum(dim=1),
                (x_ordered * x_count).sum(dim=1),
                (x_ordered * x_count_pile).sum(dim=1),
            ]
        ).tolist()
        num_valid_tokens = (
            num_cards
            + num_relics
            + num_potions
            + num_monsters
            + num_event_options
            + num_character
            + num_rooms
        )
        x_draw_order = X_DrawOrderStates(
            x_states=torch.cat(
                [
                    torch.nonzero_static(x_ordered[0], size=num_states_visible),
                    torch.nonzero_static(x_ordered[1], size=num_states_hidden),
                ]
            ).squeeze(1),
            num_states_visible=num_states_visible,
            num_rows_visible=num_rows_visible,
            num_rows=num_rows_visible + num_rows_hidden,
            num_pile_visible=num_pile_visible,
            num_pile=num_pile_visible + num_pile_hidden,
        )

        # Each window position's row in x_tokens below, whose valid tokens keep window order
        # (meaningless on a pad)
        x_token_row = (x_is_token.flatten().cumsum(0) - 1).view_as(x_is_token)  # (B, T)
        (
            x_card_row,
            x_relic_row,
            x_potion_row,
            x_monster_row,
            x_event_option_row,
            x_character_row,
            x_room_row,
        ) = x_token_row.split([x_entity.mask.shape[1] for x_entity in entity_blocks], dim=1)

        # Each block's padding dropped, its valid rows through its private first layer, and the
        # result written at their tokens' rows
        x_tokens = x_rooms.x.new_zeros(num_valid_tokens, self._dim_entity)  # (N_tok, D)
        block_projections: list[tuple[X_EntitySparse | X_Entity, nn.Linear, torch.Tensor, int]] = [
            (x_game_state.cards, self._pre_card, x_card_row, num_cards),
            (x_game_state.relics, self._pre_relic, x_relic_row, num_relics),
            (x_game_state.potions, self._pre_potion, x_potion_row, num_potions),
            (x_game_state.monsters, self._pre_monster, x_monster_row, num_monsters),
            (
                x_game_state.event_options,
                self._pre_event_option,
                x_event_option_row,
                num_event_options,
            ),
            (x_game_state.character, self._pre_character, x_character_row, num_character),
            (x_rooms, self._pre_room, x_room_row, num_rooms),
        ]
        for x_entity, mapper, x_block_token_row, num_block_tokens in block_projections:
            x_rows = _drop_padding(x_block_token_row, x_entity.mask, num_block_tokens)  # (N,)
            x_tokens[x_rows] = _first_layer(x_entity, mapper, num_block_tokens)  # (N, D)

        # Owned rows add into their owners, scaled by 1/sqrt(rows per owner), before the shared
        # ReLU/LayerNorm. Family order matters: float sums are order-sensitive
        x_slice_kind_row = _slice_kind_rows(x_slice_kind_column, x_token_row, num_valid_tokens)
        _add_owned(x_tokens, x_game_state.prices, self._pre_price, num_prices, x_slice_kind_row)
        _add_owned(
            x_tokens,
            x_game_state.health_block,
            self._pre_health_block,
            num_health_block,
            x_slice_kind_row,
        )
        _add_owned(
            x_tokens, x_game_state.effects, self._effects_mlp, num_effects, x_slice_kind_row
        )
        _add_owned(
            x_tokens, x_game_state.modifiers, self._modifiers_mlp, num_modifiers, x_slice_kind_row
        )

        # The pending halt owns no token: its effect row is a global-context input instead
        x_effect_pending = x_game_state.effect_pending
        x_pending_column = _drop_padding(
            x_effect_pending.x_column, x_effect_pending.mask, num_pending
        )  # (P, K)
        x_pending_value = _drop_padding(
            x_effect_pending.x_value, x_effect_pending.mask, num_pending
        )  # (P, K)
        x_pending_valid = _project_sparse(
            self._effects_mlp, x_pending_column, x_pending_value
        )  # (P, D)
        x_pending = _restore_padding(x_pending_valid, x_effect_pending.mask, num_pending)  # (B, D)

        # Shared second layer + norm on the valid rows
        x_hidden = self._layer_norm(self._post_shared(F.relu(x_tokens)))  # (N_tok, D)

        # The window packed: each row's state and window position pick its slice kind and masks
        x_state, x_position = torch.nonzero_static(x_is_token, size=num_valid_tokens).unbind(1)
        x_window = X_Window(
            x_slice_kind=x_slice_kind[x_state, x_position],
            x_legal=x_legal[x_state, x_position],
            x_needs_target=x_needs_target[x_state, x_position],
            x_state=x_state,
            x_position=x_position,
            x_first_row=x_count.cumsum(0) - x_count,
            x_count=x_count,
            x_slice_kind_row=x_slice_kind_row,
            x_slice_kind_count=x_slice_kind_count,
        )
        return (
            x_hidden,
            x_window,
            x_slice_kind.shape[1],
            TokenCounts(
                cards=num_cards,
                relics=num_relics,
                potions=num_potions,
                monsters=num_monsters,
                event_options=num_event_options,
                character=num_character,
                rooms=num_rooms,
            ),
            x_draw_order,
            x_pending,
        )


def _add_pooled(x_target: torch.Tensor, x_pos: torch.Tensor, x_rows: torch.Tensor) -> None:
    """Add each row into `x_target[pos]`, scaled by 1/sqrt(the rows sharing its pos)."""

    # Counted at the target's length: bincount reads max(pos) back to size its output
    x_count = torch.zeros(x_target.shape[0], dtype=torch.long, device=x_pos.device)
    x_count = x_count.index_add_(0, x_pos, torch.ones_like(x_pos))[x_pos]  # (N,)
    x_target.index_add_(0, x_pos, x_rows / x_count.to(x_rows.dtype).sqrt().unsqueeze(1))


def _drop_padding(x: torch.Tensor, x_mask: torch.Tensor, count: int) -> torch.Tensor:
    """`x`'s entries where `x_mask` is True, in order: (B, S, ...) to (count, ...) for a (B, S)
    mask, (B, ...) to (count, ...) for a (B,) one. `count` is given: nonzero would read it back
    from the GPU"""
    x_index = torch.nonzero_static(x_mask, size=count).unbind(1)
    return x[x_index]


def _restore_padding(x_valid: torch.Tensor, x_mask: torch.Tensor, count: int) -> torch.Tensor:
    """`_drop_padding`'s inverse: (count, ...) back to `x_mask`'s shape, zeros at its pads"""
    x_index = torch.nonzero_static(x_mask, size=count).unbind(1)
    x = x_valid.new_zeros(*x_mask.shape, *x_valid.shape[1:])
    x[x_index] = x_valid
    return x


class _SumSparseDenseBackward(torch.autograd.Function):
    """F.embedding_bag over the sparse entries, with the weight's gradient as one GEMM over the
    entries densified to (N, in) rows: embedding_bag's own backward sorts the entries and sums
    each column's serially"""

    @staticmethod
    def forward(
        ctx: Any, x_column: torch.Tensor, x_value: torch.Tensor, x_weight: torch.Tensor
    ) -> torch.Tensor:
        """The (N, K) entries through a linear layer's (D, in) weight, bias aside: its columns at
        the entries' columns, scaled by their values, summed"""
        ctx.save_for_backward(x_column, x_value)
        ctx.num_columns = x_weight.shape[1]

        # The transposed weight copied whole: as a strided view, a GPU reads its rows uncoalesced
        return F.embedding_bag(
            x_column.int(), x_weight.t().contiguous(), per_sample_weights=x_value, mode="sum"
        )

    @staticmethod
    def backward(ctx: Any, x_grad: torch.Tensor) -> tuple[None, None, torch.Tensor]:
        x_column, x_value = ctx.saved_tensors

        # The pad entries (0, 0.0) add zeros
        x_dense = x_grad.new_zeros(x_column.shape[0], ctx.num_columns)
        x_dense.scatter_add_(1, x_column.long(), x_value)  # (N, in)
        return None, None, x_grad.t() @ x_dense  # (D, in)


def _project_sparse(
    mapper: nn.Linear | nn.Sequential, x_column: torch.Tensor, x_value: torch.Tensor
) -> torch.Tensor:
    """`mapper` on the dense rows that the (N, K) entries stand for. Its first linear layer (the
    mapper, or a Sequential's first layer) sums its weight's columns at the entries' columns,
    scaled by their values, plus its bias; a Sequential's other layers follow"""
    linear = mapper if isinstance(mapper, nn.Linear) else mapper[0]
    x_rows = _SumSparseDenseBackward.apply(x_column, x_value, linear.weight)
    if linear.bias is not None:
        x_rows = x_rows + linear.bias
    return x_rows if isinstance(mapper, nn.Linear) else mapper[1:](x_rows)


def _first_layer(
    x_entity: X_EntitySparse | X_Entity, mapper: nn.Linear, num_valid: int
) -> torch.Tensor:
    """A block's valid tokens through its private mapper, its padding dropped: (N, D)"""
    if isinstance(x_entity, X_Entity):  # The rooms, dense from `MapGNN`
        return mapper(_drop_padding(x_entity.x, x_entity.mask, num_valid))
    x_column = _drop_padding(x_entity.x_column, x_entity.mask, num_valid)  # (N, K)
    x_value = _drop_padding(x_entity.x_value, x_entity.mask, num_valid)  # (N, K)
    return _project_sparse(mapper, x_column, x_value)


def _add_owned(
    x_tokens: torch.Tensor,
    x_owned_sparse: X_OwnedSparse,
    mapper: nn.Linear | nn.Sequential,
    num_rows: int,
    x_slice_kind_row: torch.Tensor,
) -> None:
    """Add a family's valid rows into their owners' rows of `x_tokens`. An owner is the rank-th
    token of its slice kind, whose tokens sit in consecutive rows"""
    x_state, x_slot = torch.nonzero_static(x_owned_sparse.mask, size=num_rows).unbind(1)  # (N,)
    x_rows = _project_sparse(
        mapper, x_owned_sparse.x_column[x_state, x_slot], x_owned_sparse.x_value[x_state, x_slot]
    )  # (N, D)
    x_owner_slice_kind = x_owned_sparse.x_owner_slice_kind[x_state, x_slot]  # (N,)
    x_owner_rank = x_owned_sparse.x_owner_rank[x_state, x_slot]  # (N,)
    x_owner_row = x_slice_kind_row[x_state, x_owner_slice_kind] + x_owner_rank  # (N,)
    _add_pooled(x_tokens, x_owner_row, x_rows)


def _slice_kind_rows(
    x_slice_kind_column: torch.Tensor, x_token_row: torch.Tensor, num_valid_tokens: int
) -> torch.Tensor:
    """Each state's row of each slice kind's first token, (B, len(SliceKind)):
    `num_valid_tokens` (out of range) where the kind is absent; the pads, which
    `x_slice_kind_column` puts in an extra column, are dropped. Entity blocks keep each slice
    kind's tokens contiguous (rooms have gaps, own nothing)"""
    x_slice_kind_row = x_token_row.new_full(
        (len(x_slice_kind_column), len(SliceKind) + 1), num_valid_tokens
    )
    x_slice_kind_row.scatter_reduce_(1, x_slice_kind_column, x_token_row, reduce="amin")
    return x_slice_kind_row[:, :-1]
