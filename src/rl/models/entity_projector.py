from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.encoding.card import ENCODING_DIM_CARD
from src.rl.encoding.character import ENCODING_DIM_CHARACTER
from src.rl.encoding.effect import ENCODING_DIM_EFFECT
from src.rl.encoding.event import ENCODING_DIM_EVENT_OPTION
from src.rl.encoding.health_block import ENCODING_DIM_HEALTH_BLOCK
from src.rl.encoding.modifier import ENCODING_DIM_MODIFIER
from src.rl.encoding.monster import ENCODING_DIM_MONSTER
from src.rl.encoding.potion import ENCODING_DIM_POTION
from src.rl.encoding.relic import ENCODING_DIM_RELIC
from src.rl.encoding.shop import ENCODING_DIM_PRICE
from src.rl.types import OWNER_PENDING
from src.rl.types import EntityKind
from src.rl.types import X_Entity
from src.rl.types import X_EntityPairs
from src.rl.types import X_GameState
from src.rl.types import X_Owned


def _add_pooled(x_target: torch.Tensor, x_pos: torch.Tensor, x_rows: torch.Tensor) -> None:
    """Add each row into `x_target[pos]`, scaled by 1/sqrt(the rows sharing its pos)."""

    # Counted at the target's length: bincount reads max(pos) back to size its output
    x_count = torch.zeros(x_target.shape[0], dtype=torch.long, device=x_pos.device)
    x_count = x_count.index_add_(0, x_pos, torch.ones_like(x_pos))[x_pos]  # (N,)
    x_target.index_add_(0, x_pos, x_rows / x_count.to(x_rows.dtype).sqrt().unsqueeze(1))


def _positions(x_mask: torch.Tensor, count: int) -> torch.Tensor:
    """The positions where the 1-D `x_mask` is True, given how many: nonzero would read the
    count back from the GPU"""
    return torch.nonzero_static(x_mask, size=count).squeeze(1)


class _SumPairsDenseBackward(torch.autograd.Function):
    """F.embedding_bag over the pairs, with the weight's gradient as one GEMM over the pairs
    densified to (N, in) rows: embedding_bag's own backward sorts the pairs and sums each
    column's serially"""

    @staticmethod
    def forward(
        ctx: Any, x_column: torch.Tensor, x_value: torch.Tensor, x_weight: torch.Tensor
    ) -> torch.Tensor:
        """The (N, K) pairs through a linear layer's (D, in) weight, bias aside: its columns at
        the pairs' columns, scaled by their values, summed"""
        ctx.save_for_backward(x_column, x_value)
        ctx.num_columns = x_weight.shape[1]

        # The transposed weight copied whole: as a strided view, a GPU reads its rows uncoalesced
        return F.embedding_bag(
            x_column.int(), x_weight.t().contiguous(), per_sample_weights=x_value, mode="sum"
        )

    @staticmethod
    def backward(ctx: Any, x_grad: torch.Tensor) -> tuple[None, None, torch.Tensor]:
        x_column, x_value = ctx.saved_tensors

        # The pad pairs (0, 0.0) add zeros
        x_dense = x_grad.new_zeros(x_column.shape[0], ctx.num_columns)
        x_dense.scatter_add_(1, x_column.long(), x_value)  # (N, in)
        return None, None, x_grad.t() @ x_dense  # (D, in)


def _project_pairs(
    mapper: nn.Linear | nn.Sequential, x_column: torch.Tensor, x_value: torch.Tensor
) -> torch.Tensor:
    """`mapper` on the dense rows that the (N, K) pairs stand for. Its first linear layer (the
    mapper, or a Sequential's first layer) sums its weight's columns at the pairs' columns,
    scaled by their values, plus its bias; a Sequential's other layers follow"""
    linear = mapper if isinstance(mapper, nn.Linear) else mapper[0]
    x_rows = _SumPairsDenseBackward.apply(x_column, x_value, linear.weight)
    if linear.bias is not None:
        x_rows = x_rows + linear.bias
    return x_rows if isinstance(mapper, nn.Linear) else mapper[1:](x_rows)


def _project_kind(
    x_entity: X_EntityPairs | X_Entity,
    mapper: nn.Linear,
    count: int,
    x_block_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One entity kind's `count` valid rows through its private mapper, and their rows in the
    window's valid tokens, read from `x_block_pos`: the kind's (B, W) columns of the window's
    `x_token_pos`"""

    # Calculate non-padding tokens
    x_row = _positions(torch.flatten(x_entity.mask), count)

    # Forward pass
    if isinstance(x_entity, X_Entity):  # The rooms, dense from `MapGNN`
        x_first = mapper(x_entity.x.flatten(0, 1)[x_row])
    else:
        x_first = _project_pairs(
            mapper,
            x_entity.x_column.flatten(0, 1)[x_row],
            x_entity.x_value.flatten(0, 1)[x_row],
        )
    return x_first, x_block_pos.flatten()[x_row]


class EntityProjector(nn.Module):
    """Factored (AlphaStar-style) entity embedder: a private first layer per kind, the shared
    row families (health/block, prices, effects, modifiers) added into their owners, then one
    shared second layer + LayerNorm. Returns the window as ONE X_Entity (the six blocks then
    the rooms) and the pooled pending halt."""

    def __init__(self, dim_entity: int, dim_map: int) -> None:
        super().__init__()

        self._dim_entity = dim_entity

        # Per-`EntityKind` private mappers
        self._pre_card = nn.Linear(ENCODING_DIM_CARD, dim_entity)
        self._pre_relic = nn.Linear(ENCODING_DIM_RELIC, dim_entity)
        self._pre_potion = nn.Linear(ENCODING_DIM_POTION, dim_entity)
        self._pre_event_option = nn.Linear(ENCODING_DIM_EVENT_OPTION, dim_entity)
        self._pre_monster = nn.Linear(ENCODING_DIM_MONSTER, dim_entity)
        self._pre_character = nn.Linear(ENCODING_DIM_CHARACTER, dim_entity)
        self._pre_room = nn.Linear(dim_map, dim_entity)  # Rooms come already-processed by `MapGNN`

        # Health+Block's mapper is shared between the Character and the Monsters
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

        # Shared second layer (after ReLU, on the fused all-kinds row tensor)
        self._post_shared = nn.Linear(dim_entity, dim_entity)

        self._layer_norm = nn.LayerNorm(dim_entity)

    def forward(
        self, x_game_state: X_GameState, x_rooms: X_Entity
    ) -> tuple[X_Entity, torch.Tensor]:
        # Shapes: B = batch, D = embedding width, T = the window's tokens, N = a family's
        # valid rows across the batch, N_tok = the window's valid tokens
        entity_kinds: list[tuple[X_EntityPairs | X_Entity, nn.Linear]] = [
            (x_game_state.cards, self._pre_card),
            (x_game_state.relics, self._pre_relic),
            (x_game_state.potions, self._pre_potion),
            (x_game_state.monsters, self._pre_monster),
            (x_game_state.event_options, self._pre_event_option),
            (x_game_state.character, self._pre_character),
            (x_rooms, self._pre_room),
        ]

        # EntityKind order: the list above IS the window's entity order, the order
        # every consumer cats the kind rows in

        # The owned-row families, in the order they add into their owners
        families: list[tuple[X_Owned, nn.Linear | nn.Sequential]] = [
            (x_game_state.prices, self._pre_price),
            (x_game_state.health_block, self._pre_health_block),
            (x_game_state.effects, self._effects_mlp),
            (x_game_state.modifiers, self._modifiers_mlp),
        ]
        x_slice_kind = torch.cat(
            [x_entity.x_slice_kind for x_entity, _ in entity_kinds], dim=1
        )  # (B, T)

        # Every pack's size in one read: the CPU waits for the GPU here, and nowhere else below
        counts = torch.stack(
            [x_entity.mask.sum() for x_entity, _ in entity_kinds]
            + [x_owned.mask.sum() for x_owned, _ in families]
        ).tolist()
        valid_counts = counts[: len(entity_kinds)]
        owned_counts = counts[len(entity_kinds) :]
        num_valid = sum(valid_counts)
        batch_size, num_tokens = x_slice_kind.shape
        device = x_slice_kind.device
        x_mask = (x_slice_kind >= 0).flatten()  # (B * T,) bool

        # Each window position's row in x_tokens below, whose valid tokens keep window order
        # (meaningless on a pad)
        x_token_pos = (x_mask.cumsum(0) - 1).view(batch_size, num_tokens)  # (B, T)
        widths = [x_entity.x_slice_kind.shape[1] for x_entity, _ in entity_kinds]
        x_firsts, x_firsts_pos = zip(
            *[
                _project_kind(x_entity, mapper, count, x_block_pos)
                for (x_entity, mapper), count, x_block_pos in zip(
                    entity_kinds, valid_counts, x_token_pos.split(widths, dim=1)
                )
            ]
        )

        # The valid tokens, then one pending-halt slot per state
        x_first = torch.cat(x_firsts)  # (N_tok, D)
        x_tokens = x_first.new_zeros(num_valid + batch_size, self._dim_entity)  # (N_tok + B, D)
        x_tokens[torch.cat(x_firsts_pos)] = x_first

        # Each EntityKind's first window column: the widths of the blocks before it
        # The halt's entry, 0, keeps its rows' window positions in range
        x_start = torch.tensor(
            [sum(widths[:kind]) for kind in EntityKind] + [0], pin_memory=device.type == "cuda"
        )
        x_start = x_start.to(device, non_blocking=True)  # pinned, so the CPU doesn't wait

        # Owned rows add into their owners, scaled by 1/sqrt(rows per owner), before the shared
        # ReLU/LayerNorm. Family order matters: float sums are order-sensitive
        for (x_owned, mapper), owned in zip(families, owned_counts):
            x_batch, x_slot = torch.nonzero_static(x_owned.mask, size=owned).unbind(1)  # (N,)
            x_rows = _project_pairs(
                mapper,
                x_owned.x_column[x_batch, x_slot],
                x_owned.x_value[x_batch, x_slot],
            )  # (N, D)
            x_kind = x_owned.x_entity_kind[x_batch, x_slot]  # (N,)
            x_offset = x_owned.x_offset[x_batch, x_slot]  # (N,)
            x_pos = x_token_pos[x_batch, x_start[x_kind] + x_offset]  # (N,)

            # The halt owns no entity, so it becomes a global-context input instead
            x_pos = torch.where(x_kind == OWNER_PENDING, num_valid + x_batch, x_pos)
            _add_pooled(x_tokens, x_pos, x_rows)

        # Shared second layer + norm on the valid rows only, so pads stay exact zeros
        x_valid, x_pending = x_tokens.split([num_valid, batch_size])
        x_hidden = self._layer_norm(self._post_shared(F.relu(x_valid)))  # (N_tok, D)

        x_row = _positions(x_mask, num_valid)  # (N_tok,)
        x_entities = torch.zeros(
            batch_size * num_tokens, self._dim_entity, device=device
        )  # (B * T, D)
        x_entities[x_row] = x_hidden

        # The window and its kind row (long: the model indexes with it)
        x_window = X_Entity(
            x_entities.reshape(batch_size, num_tokens, self._dim_entity), x_slice_kind.long()
        )  # (B, T, D) tokens + their (B, T) kind row
        return x_window, x_pending
