from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import flex_attention

from src.rl.constants import MAX_SIZE_CARD_PILE_DRAW
from src.rl.types import SliceKind
from src.rl.types import X_Window


# FlexAttention's block: on the GPU, each side of an attention is padded to a multiple of it
_FLEX_BLOCK = 128

# Fixed kernel options: max-autotune leaves the forward's untuned, and tunes only the backward's
# block sizes
_FLEX_KERNEL_OPTIONS = {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 3}

# On the GPU even the eager actor calls FlexAttention compiled: eager, it scores every query row
# against every key row. One graph, or a loud failure rather than a fall back to eager
_flex_attention_compiled = torch.compile(flex_attention, dynamic=True, fullgraph=True)


def _padded(num_rows: int, device: torch.device) -> int:
    """`num_rows` rounded up to a multiple of FlexAttention's block on the GPU; on the CPU, where
    SDPA needs no tail, `num_rows`"""
    if device.type != "cuda":
        return num_rows
    return -(-num_rows // _FLEX_BLOCK) * _FLEX_BLOCK


def _states_and_ranks(x_offsets: torch.Tensor, num_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Each of `num_rows` rows' state and its rank in it, the states laid out one after another
    from `x_offsets` (S + 1,), their first rows then the end: the rows past the end form state S"""
    x_row = torch.arange(num_rows, device=x_offsets.device)
    x_state = torch.searchsorted(x_offsets, x_row, right=True) - 1
    return x_state, x_row - x_offsets[x_state]


def _rows_by_state(x_window: X_Window) -> torch.Tensor:
    """The window's rows laid out by state: each state's rows, state after state, (N,). The sort
    is stable, so each state's rows keep their block order"""
    return torch.argsort(x_window.x_state, stable=True)


def _block_mask(
    x_offsets_query: torch.Tensor, x_offsets_key: torch.Tensor, num_query: int, num_key: int
) -> BlockMask:
    """FlexAttention's same-state mask, built on the device: each query row attends the key rows
    of its own state. `x_offsets_*` (S + 1,) are each side's state starts, then its end; its
    `num_*` rows, a multiple of _FLEX_BLOCK, end in a tail that forms state S"""
    x_state_query, _ = _states_and_ranks(x_offsets_query, num_query)
    x_state_key, _ = _states_and_ranks(x_offsets_key, num_key)

    # A query block attends the key blocks from its first row's state's first key row to its last
    # row's state's last key row
    num_blocks_key = num_key // _FLEX_BLOCK
    x_starts_key = F.pad(x_offsets_key, (0, 1), value=num_key)  # (S + 2,)
    x_first = x_starts_key[x_state_query[::_FLEX_BLOCK]] // _FLEX_BLOCK
    x_last = (x_starts_key[x_state_query[_FLEX_BLOCK - 1 :: _FLEX_BLOCK] + 1] - 1) // _FLEX_BLOCK

    # from_kv_blocks wants each query block's key blocks as wide as the key blocks: its first
    # (x_last - x_first + 1) count
    x_blocks_key = x_first.unsqueeze(1) + torch.arange(num_blocks_key, device=x_first.device)

    def same_state(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        return x_state_query[q_idx] == x_state_key[kv_idx]

    return BlockMask.from_kv_blocks(
        (x_last - x_first + 1).int().view(1, 1, -1),
        x_blocks_key.clamp(max=num_blocks_key - 1).int().view(1, 1, -1, num_blocks_key),
        BLOCK_SIZE=_FLEX_BLOCK,
        mask_mod=same_state,
        seq_lengths=(num_query, num_key),
    )


def _same_state_mask(
    x_offsets_query: torch.Tensor, x_offsets_key: torch.Tensor, num_query: int, num_key: int
) -> BlockMask | torch.Tensor | None:
    """Each query row attends the key rows of its own state, `x_offsets_*` (S + 1,) each side's
    state starts, then its end: on the GPU FlexAttention's BlockMask; on the CPU SDPA's dense
    (N_q, N_k) mask, or none for a single state"""
    if x_offsets_query.is_cuda:
        return _block_mask(x_offsets_query, x_offsets_key, num_query, num_key)
    if x_offsets_query.shape[0] == 2:
        return None
    x_state_query, _ = _states_and_ranks(x_offsets_query, num_query)
    x_state_key, _ = _states_and_ranks(x_offsets_key, num_key)
    return x_state_query.unsqueeze(1) == x_state_key.unsqueeze(0)


def _attention(
    mha: nn.MultiheadAttention,
    x_query: torch.Tensor,
    x_key: torch.Tensor,
    mask: BlockMask | torch.Tensor | None,
) -> torch.Tensor:
    """`mha` over packed rows, its own projections around the attention kernel: each query row
    (N_q, D) reads the key rows (N_k, D) that `mask` lets it. FlexAttention, compiled, on the GPU;
    SDPA on the CPU"""
    dim = mha.embed_dim
    x_q = F.linear(x_query, mha.in_proj_weight[:dim], mha.in_proj_bias[:dim])
    x_k, x_v = F.linear(x_key, mha.in_proj_weight[dim:], mha.in_proj_bias[dim:]).chunk(2, dim=1)

    # Both kernels' layout: (1, heads, rows, head dim)
    x_q, x_k, x_v = (
        x.unflatten(1, (mha.num_heads, -1)).transpose(0, 1).unsqueeze(0) for x in (x_q, x_k, x_v)
    )

    # One tensor, (1, H, N_q, D / H): no log-sum-exp is asked for
    if isinstance(mask, BlockMask):
        x_read = cast(
            torch.Tensor,
            _flex_attention_compiled(
                x_q, x_k, x_v, block_mask=mask, kernel_options=_FLEX_KERNEL_OPTIONS
            ),
        )
    else:
        x_read = F.scaled_dot_product_attention(x_q, x_k, x_v, attn_mask=mask)
    return mha.out_proj(x_read.squeeze(0).transpose(0, 1).flatten(1))


class _EntityTransformerBlock(nn.Module):
    def __init__(self, dim_entity: int, dim_feed_forward: int, num_heads: int) -> None:
        super().__init__()

        self._multi_head_attention = nn.MultiheadAttention(dim_entity, num_heads, batch_first=True)
        self._layer_norm_1 = nn.LayerNorm(dim_entity)
        self._layer_norm_2 = nn.LayerNorm(dim_entity)
        self._mlp = nn.Sequential(
            nn.Linear(dim_entity, dim_feed_forward),
            nn.ReLU(),
            nn.Linear(dim_feed_forward, dim_entity),
        )

    def forward(
        self, x_packed: torch.Tensor, mask: BlockMask | torch.Tensor | None
    ) -> torch.Tensor:
        # Pre-LN (norm inside the residual branch): stable at depth, unlike post-LN
        x_normed = self._layer_norm_1(x_packed)
        x_mha = _attention(self._multi_head_attention, x_normed, x_normed, mask)
        x_out = x_packed + x_mha

        # Feedforward with residual connection
        x_out = x_out + self._mlp(self._layer_norm_2(x_out))

        return x_out


class EntityTransformer(nn.Module):
    def __init__(
        self, dim_entity: int, dim_feed_forward: int, num_heads: int, num_blocks: int
    ) -> None:
        super().__init__()

        self._entity_transformer_blocks = nn.ModuleList(
            [
                _EntityTransformerBlock(dim_entity, dim_feed_forward, num_heads)
                for _ in range(num_blocks)
            ]
        )

        # Pre-LN blocks leave the residual stream unnormalized; one final norm
        self._norm_final = nn.LayerNorm(dim_entity)

        # Learned global token, refined with the entities (replaces pooling). It
        # has no slice kind, so it joins the sequence here and leaves it here
        self._entity_global = nn.Parameter(torch.empty(1, 1, dim_entity))
        nn.init.normal_(self._entity_global, std=0.02)

    def forward(
        self, x_tokens: torch.Tensor, x_window: X_Window
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Refine the packed window's rows (N, D), in block order, together with each state's
        global token, which joins the sequence at the end of its state's rows.

        Returns the refined rows (N, D), in block order, and global tokens (B, D)."""
        num_rows, dim = x_tokens.shape
        batch_size = x_window.x_count.shape[0]

        # The sequence: each state's rows, then its global token, state after state. Each state's
        # first place in it, then the end; each row's place, its position by state past one global
        # token per state before its own, and each global token's
        x_offsets = F.pad((x_window.x_count + 1).cumsum(0), (1, 0))  # (B + 1,)
        x_row_by_state = _rows_by_state(x_window)  # (N,)
        x_place_row = torch.empty_like(x_row_by_state)
        x_place_row[x_row_by_state] = (
            torch.arange(num_rows, device=x_tokens.device) + x_window.x_state[x_row_by_state]
        )  # (N,)
        x_place_global = x_offsets[1:] - 1  # (B,)

        # On the GPU, a tail of zero rows pads it to FlexAttention's block. Zeros, not empty: a
        # key's NaN would reach every query through its zero attention weight (0 * NaN = NaN)
        num_places = _padded(num_rows + batch_size, x_tokens.device)
        x_sequence = x_tokens.new_zeros(num_places, dim)
        x_sequence[x_place_row] = x_tokens
        x_sequence[x_place_global] = self._entity_global.view(1, dim)
        x_refined = self._refine(
            x_sequence, _same_state_mask(x_offsets, x_offsets, num_places, num_places)
        )
        return x_refined[x_place_row], x_refined[x_place_global]

    def _refine(
        self, x_packed: torch.Tensor, mask: BlockMask | torch.Tensor | None
    ) -> torch.Tensor:
        """The blocks, then the final norm, over packed rows"""
        for entity_transformer_block in self._entity_transformer_blocks:
            x_packed = entity_transformer_block(x_packed, mask)
        return self._norm_final(x_packed)


class _DrawOrderBlock(nn.Module):
    def __init__(self, dim_entity: int, dim_feed_forward: int, num_heads: int) -> None:
        super().__init__()

        # One row per depth in the draw pile: the cards above the token
        self._draw_depth_emb = nn.Embedding(MAX_SIZE_CARD_PILE_DRAW, dim_entity)

        self._cross_attention = nn.MultiheadAttention(dim_entity, num_heads, batch_first=True)
        self._readout_attention = nn.MultiheadAttention(dim_entity, num_heads, batch_first=True)
        self._layer_norm_1 = nn.LayerNorm(dim_entity)
        self._layer_norm_2 = nn.LayerNorm(dim_entity)
        self._layer_norm_3 = nn.LayerNorm(dim_entity)
        self._mlp = nn.Sequential(
            nn.Linear(dim_entity, dim_feed_forward),
            nn.ReLU(),
            nn.Linear(dim_feed_forward, dim_entity),
        )

        # Pre-LN leaves the residual stream unnormalized: one final norm, shared by the tokens and
        # the global token, as the transformer's
        self._norm_final = nn.LayerNorm(dim_entity)

    def forward(
        self,
        x_packed: torch.Tensor,
        x_pile: torch.Tensor,
        x_depth: torch.Tensor,
        x_global_token: torch.Tensor,
        mask_pile: BlockMask | torch.Tensor | None,
        mask_readout: BlockMask | torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Over packed rows (N, D): every row reads its state's pile (P, D), tagged with its
        depths (P,), then each state's global token (S, D) reads its state's order-aware rows.
        Pre-LN and residual, then one final norm over both outputs, as in the transformer"""
        # The pile's tokens, tagged with their depths
        x_pile = self._layer_norm_1(x_pile + self._draw_depth_emb(x_depth))

        # Every token reads the pile, then the feedforward
        x_read = _attention(self._cross_attention, self._layer_norm_1(x_packed), x_pile, mask_pile)
        x_out = x_packed + x_read
        x_out = x_out + self._mlp(self._layer_norm_2(x_out))

        # The global token reads the order-aware tokens
        x_normed = self._layer_norm_3(x_out)
        x_read_global = _attention(
            self._readout_attention,
            self._layer_norm_3(x_global_token),
            x_normed,
            mask_readout,
        )
        return self._norm_final(x_out), self._norm_final(x_global_token + x_read_global)


class DrawOrderLayer(nn.Module):
    def __init__(self, dim_entity: int, dim_feed_forward: int, num_heads: int) -> None:
        super().__init__()

        self._draw_order_block = _DrawOrderBlock(dim_entity, dim_feed_forward, num_heads)

    def forward(
        self,
        x_tokens: torch.Tensor,
        x_global_token: torch.Tensor,
        x_window: X_Window,
        x_states: torch.Tensor,
        num_rows: int,
        num_pile: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The layer over the states `x_states` (S,), each with an ordered draw pile, of the
        transformer's refined rows (N, D), in block order, and global tokens (B, D); `num_rows`
        and `num_pile` are those states' rows and draw tokens.

        Returns those states' rows in the window (num_rows,), state after state, each state's in
        block order, their order-aware rows (num_rows, D), and the states' global tokens (S, D)."""
        num_states = x_states.shape[0]

        # The states' rows, state after state: each state's first among them, then the end. Each
        # is the row `_rows_by_state` puts at its state's first position plus its rank
        x_offsets = F.pad(x_window.x_count[x_states].cumsum(0), (1, 0))  # (S + 1,)
        x_state_layer, x_rank = _states_and_ranks(x_offsets, num_rows)  # (num_rows,)
        x_first_by_state = x_window.x_count.cumsum(0) - x_window.x_count  # (B,)
        x_rows = _rows_by_state(x_window)[x_first_by_state[x_states][x_state_layer] + x_rank]

        # Their piles' rows the same way, each pile's first row plus its rank, bottom first; a draw
        # token's depth is the cards above it, its pile's count - 1 - its rank
        x_count_pile = x_window.x_slice_kind_count[x_states, SliceKind.CARD_PILE_DRAW]  # (S,)
        x_offsets_pile = F.pad(x_count_pile.cumsum(0), (1, 0))  # (S + 1,)
        x_state_layer_pile, x_rank_pile = _states_and_ranks(x_offsets_pile, num_pile)
        x_first_row_pile = x_window.x_slice_kind_row[x_states, SliceKind.CARD_PILE_DRAW]  # (S,)
        x_rows_pile = x_first_row_pile[x_state_layer_pile] + x_rank_pile
        x_depth = x_count_pile[x_state_layer_pile] - 1 - x_rank_pile

        # On the GPU, each side padded to FlexAttention's block; a global token is its state's one
        # query
        num_places = _padded(num_rows, x_tokens.device)
        num_places_pile = _padded(num_pile, x_tokens.device)
        num_places_global = _padded(num_states, x_tokens.device)
        x_offsets_global = torch.arange(num_states + 1, device=x_states.device)
        x_out, x_out_global = self._draw_order_block(
            F.pad(x_tokens[x_rows], (0, 0, 0, num_places - num_rows)),
            F.pad(x_tokens[x_rows_pile], (0, 0, 0, num_places_pile - num_pile)),
            F.pad(x_depth, (0, num_places_pile - num_pile)),
            F.pad(x_global_token[x_states], (0, 0, 0, num_places_global - num_states)),
            _same_state_mask(x_offsets, x_offsets_pile, num_places, num_places_pile),
            _same_state_mask(x_offsets_global, x_offsets, num_places_global, num_places),
        )
        return x_rows, x_out[:num_rows], x_out_global[:num_states]
