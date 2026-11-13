from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import flex_attention

from src.rl.constants import MAX_SIZE_CARD_PILE_DRAW
from src.rl.types import Jagged
from src.rl.types import TokenKind


# FlexAttention's block: on the GPU, each side of an attention is padded to a multiple of it
_FLEX_BLOCK = 128

# Fixed kernel options: max-autotune leaves the forward's untuned, and tunes only the backward
_FLEX_KERNEL_OPTIONS = {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 3}

# On the GPU even the eager Actor calls FlexAttention compiled
_flex_attention_compiled = torch.compile(flex_attention, dynamic=True, fullgraph=True)


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
        self._layer_norm_final = nn.LayerNorm(dim_entity)

        # Learned global token, refined with the Entities
        self._x_entity_global = nn.Parameter(torch.empty(dim_entity))
        nn.init.normal_(self._x_entity_global, std=0.02)

    def forward(
        self, x_tokens: Jagged[torch.Tensor], x_state: torch.Tensor
    ) -> tuple[Jagged[torch.Tensor], torch.Tensor]:
        """Refine the packed window's rows together with each state's global token.

        Each state's global token joins the sequence at the end of its state's rows.

        Parameters
        ----------
        x_tokens : Jagged[torch.Tensor]
            (B,) lengths and (N, D) rows, one segment per state.
        x_state : torch.Tensor
            (N,) long. The cached state index for each row.

        Returns
        -------
        refined : Jagged[torch.Tensor]
            The refined rows, with the same lengths and order as the input.
        x_global_token : torch.Tensor
            (B, D). The global tokens.
        """
        x_rows = x_tokens.rows
        num_rows, dim = x_rows.shape
        batch_size = x_tokens.x_lengths.shape[0]

        # Start row of each state in `x_rows`. The +1 is for the states' global token
        x_off = F.pad((x_tokens.x_lengths + 1).cumsum(0), (1, 0))  # (B + 1,)

        # Calculate number of FlexAttention slots and initialize tensor to hold the sequence
        num_slots = _padded(num_rows + batch_size, x_rows.device)
        x_sequence = torch.zeros(num_slots, dim, dtype=x_rows.dtype, device=x_rows.device)

        # Calculate each token's position in the sequence. The global token goes last in each state's sequence
        x_place_row = x_state + torch.arange(num_rows, device=x_rows.device)   # (num_rows,)
        x_place_global = x_off[1:] - 1  # (B,)

        # Fill the empty sequence tensor
        x_sequence[x_place_row] = x_rows
        x_sequence[x_place_global] = self._x_entity_global

        # Run Attention
        x_refined = self._refine(
            x_sequence, _same_state_mask(x_off, x_off, num_slots, num_slots)
        )

        # Return refined tokens and each state's global token
        return (
            Jagged(x_lengths=x_tokens.x_lengths, rows=x_refined[x_place_row]),
            x_refined[x_place_global],
        )

    def _refine(
        self, x_packed: torch.Tensor, x_mask: BlockMask | torch.Tensor | None
    ) -> torch.Tensor:
        """Run the blocks, then the final norm, over packed rows.

        Parameters
        ----------
        x_packed : torch.Tensor
            (N, D). The packed rows.
        x_mask : BlockMask | torch.Tensor | None
            The same-state mask the blocks' attention reads.

        Returns
        -------
        torch.Tensor
            (N, D). The refined rows.
        """
        for entity_transformer_block in self._entity_transformer_blocks:
            x_packed = entity_transformer_block(x_packed, x_mask)

        return self._layer_norm_final(x_packed)


class _DrawOrderBlock(nn.Module):
    def __init__(self, dim_entity: int, dim_feed_forward: int, num_heads: int) -> None:
        super().__init__()

        # One row per depth in the draw pile: the Cards above the token
        self._draw_depth_emb = nn.Embedding(MAX_SIZE_CARD_PILE_DRAW, dim_entity)

        self._cross_attention = _Attention(dim_entity, num_heads)
        self._readout_attention = _Attention(dim_entity, num_heads)
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
        self._layer_norm_final = nn.LayerNorm(dim_entity)

    def forward(
        self,
        x_packed: torch.Tensor,
        x_pile: torch.Tensor,
        x_depth: torch.Tensor,
        x_global_token: torch.Tensor,
        x_mask_pile: BlockMask | torch.Tensor | None,
        x_mask_readout: BlockMask | torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the draw-order block over packed rows.

        Every row reads its state's pile, tagged with its depths, then each state's global token
        reads its state's order-aware rows. Pre-LN and residual, then one final norm over both
        outputs, as in the transformer.

        Parameters
        ----------
        x_packed : torch.Tensor
            (N, D). The packed rows.
        x_pile : torch.Tensor
            (P, D). The piles' rows: the states' draw tokens.
        x_depth : torch.Tensor
            (P,) long. Each pile row's depth: the Cards above it.
        x_global_token : torch.Tensor
            (S, D). Each state's global token.
        x_mask_pile : BlockMask | torch.Tensor | None
            The mask that lets each row read its state's pile.
        x_mask_readout : BlockMask | torch.Tensor | None
            The mask that lets each global token read its state's rows.

        Returns
        -------
        x_out : torch.Tensor
            (N, D). The order-aware rows.
        x_out_global : torch.Tensor
            (S, D). The states' global tokens.
        """
        # The pile's tokens, tagged with their depths
        x_pile = self._layer_norm_1(x_pile + self._draw_depth_emb(x_depth))

        # Every token reads the pile, then the feedforward
        x_read = self._cross_attention(self._layer_norm_1(x_packed), x_pile, x_mask_pile)
        x_out = x_packed + x_read
        x_out = x_out + self._mlp(self._layer_norm_2(x_out))

        # The global token reads the order-aware tokens
        x_normed = self._layer_norm_3(x_out)
        x_read_global = self._readout_attention(
            self._layer_norm_3(x_global_token),
            x_normed,
            x_mask_readout,
        )
        return self._layer_norm_final(x_out), self._layer_norm_final(x_global_token + x_read_global)


class DrawOrderLayer(nn.Module):
    def __init__(self, dim_entity: int, dim_feed_forward: int, num_heads: int) -> None:
        super().__init__()

        self._draw_order_block = _DrawOrderBlock(dim_entity, dim_feed_forward, num_heads)

    def forward(
        self,
        refined: Jagged[torch.Tensor],
        x_global_token: torch.Tensor,
        x_token_kind_count: torch.Tensor,
        x_states: torch.Tensor,
        num_rows: int,
        num_pile: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the layer over the states `x_states`, each with an ordered draw pile.

        Parameters
        ----------
        refined : Jagged[torch.Tensor]
            The transformer's refined rows, one segment per state.
        x_global_token : torch.Tensor
            (B, D). The transformer's global tokens.
        x_token_kind_count : torch.Tensor
            (B, len(TokenKind)) int16. Each state's tokens of each token kind.
        x_states : torch.Tensor
            (S,) long. The states, each with an ordered draw pile.
        num_rows : int
            Those states' number of tokens.
        num_pile : int
            Those states' number of draw tokens.

        Returns
        -------
        x_rows : torch.Tensor
            (num_rows,) long. Those states' rows, state after state, each state's in window
            order.
        x_refined_layer : torch.Tensor
            (num_rows, D). Their order-aware rows.
        x_global_token_layer : torch.Tensor
            (S, D). The states' global tokens.
        """
        batch_size = x_states.shape[0]

        # The states' rows, state after state: each state's first among them, then the end. Each
        # is its state's first row in the window plus its rank
        x_off = F.pad(refined.x_lengths[x_states].cumsum(0), (1, 0))  # (S + 1,)
        x_state_layer, x_rank = _states_and_ranks(x_off, num_rows)  # (num_rows,)
        x_state_row = refined.x_lengths.cumsum(0) - refined.x_lengths  # (B,)
        x_rows = x_state_row[x_states][x_state_layer] + x_rank

        # Their piles' rows the same way, each pile's first row plus its rank, bottom first; a draw
        # token's depth is the Cards above it, its pile's count - 1 - its rank
        x_count_pile = x_token_kind_count[x_states, TokenKind.CARD_PILE_DRAW]  # (S,)
        x_off_pile = F.pad(x_count_pile.cumsum(0), (1, 0))  # (S + 1,)
        x_state_layer_pile, x_rank_pile = _states_and_ranks(x_off_pile, num_pile)

        # A pile's first row: its state's first row, past the token kinds before the draw pile's
        x_count_before_pile = x_token_kind_count[:, : TokenKind.CARD_PILE_DRAW].sum(1)  # (B,)
        x_first_row_pile = (x_state_row + x_count_before_pile)[x_states]  # (S,)
        x_rows_pile = x_first_row_pile[x_state_layer_pile] + x_rank_pile
        x_depth = x_count_pile[x_state_layer_pile] - 1 - x_rank_pile

        # On the GPU, each side padded to FlexAttention's block; a global token is its state's one
        # query
        num_places = _padded(num_rows, refined.rows.device)
        num_places_pile = _padded(num_pile, refined.rows.device)
        num_places_global = _padded(batch_size, refined.rows.device)
        x_off_global = torch.arange(batch_size + 1, device=x_states.device)
        x_out, x_out_global = self._draw_order_block(
            F.pad(refined.rows[x_rows], (0, 0, 0, num_places - num_rows)),
            F.pad(refined.rows[x_rows_pile], (0, 0, 0, num_places_pile - num_pile)),
            F.pad(x_depth, (0, num_places_pile - num_pile)),
            F.pad(x_global_token[x_states], (0, 0, 0, num_places_global - batch_size)),
            _same_state_mask(x_off, x_off_pile, num_places, num_places_pile),
            _same_state_mask(x_off_global, x_off, num_places_global, num_places),
        )
        return x_rows, x_out[:num_rows], x_out_global[:batch_size]


def _padded(num_rows: int, device: torch.device) -> int:
    """Round `num_rows` up to a multiple of FlexAttention's block on the GPU.

    Parameters
    ----------
    num_rows : int
        The number of rows.
    device : torch.device
        The rows' device.

    Returns
    -------
    int
        The padded number of rows on the GPU; on the CPU, where SDPA needs no tail, `num_rows`.
    """
    if device.type != "cuda":
        return num_rows
    
    return -(-num_rows // _FLEX_BLOCK) * _FLEX_BLOCK


def _states_and_ranks(x_off: torch.Tensor, num_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Find each of `num_rows` rows' state and its rank in it.

    The states are laid out one after another; the rows past the end form state S.

    Parameters
    ----------
    x_off : torch.Tensor
        (S + 1,) long. The states' first rows, then the end.
    num_rows : int
        The number of rows.

    Returns
    -------
    x_state : torch.Tensor
        (num_rows,) long. Each row's state.
    x_rank : torch.Tensor
        (num_rows,) long. Each row's rank in its state.
    """
    x_row = torch.arange(num_rows, device=x_off.device)
    x_state = torch.searchsorted(x_off, x_row, right=True) - 1
    return x_state, x_row - x_off[x_state]


def _block_mask(
    x_off_query: torch.Tensor, x_off_key: torch.Tensor, num_query: int, num_key: int
) -> BlockMask:
    """Build FlexAttention's same-state mask on the device.

    Each query row attends the key rows of its own state.

    Parameters
    ----------
    x_off_query : torch.Tensor
        (S + 1,) long. The query side's state starts, then its end.
    x_off_key : torch.Tensor
        (S + 1,) long. The key side's state starts, then its end.
    num_query : int
        The query side's rows, a multiple of _FLEX_BLOCK, which end in a tail that forms state S.
    num_key : int
        The key side's rows, a multiple of _FLEX_BLOCK, which end in a tail that forms state S.

    Returns
    -------
    BlockMask
        The same-state mask.
    """
    x_state_query, _ = _states_and_ranks(x_off_query, num_query)
    x_state_key, _ = _states_and_ranks(x_off_key, num_key)

    # A query block attends the key blocks from its first row's state's first key row to its last
    # row's state's last key row
    num_blocks_key = num_key // _FLEX_BLOCK
    x_starts_key = F.pad(x_off_key, (0, 1), value=num_key)  # (S + 2,)
    x_first = x_starts_key[x_state_query[::_FLEX_BLOCK]] // _FLEX_BLOCK
    x_last = (x_starts_key[x_state_query[_FLEX_BLOCK - 1 :: _FLEX_BLOCK] + 1] - 1) // _FLEX_BLOCK

    # from_kv_blocks wants each query block's key blocks as wide as the key blocks: its first
    # (x_last - x_first + 1) count
    x_blocks_key = x_first.unsqueeze(1) + torch.arange(num_blocks_key, device=x_first.device)

    def same_state(
        x_b: torch.Tensor, x_h: torch.Tensor, x_q_idx: torch.Tensor, x_kv_idx: torch.Tensor
    ) -> torch.Tensor:
        return x_state_query[x_q_idx] == x_state_key[x_kv_idx]

    return BlockMask.from_kv_blocks(
        (x_last - x_first + 1).int().view(1, 1, -1),
        x_blocks_key.clamp(max=num_blocks_key - 1).int().view(1, 1, -1, num_blocks_key),
        BLOCK_SIZE=_FLEX_BLOCK,
        mask_mod=same_state,
        seq_lengths=(num_query, num_key),
    )


def _same_state_mask(
    x_off_query: torch.Tensor, x_off_key: torch.Tensor, num_query: int, num_key: int
) -> BlockMask | torch.Tensor | None:
    """Build the mask that lets each query row attend the key rows of its own state.

    Parameters
    ----------
    x_off_query : torch.Tensor
        (S + 1,) long. The query side's state starts, then its end.
    x_off_key : torch.Tensor
        (S + 1,) long. The key side's state starts, then its end.
    num_query : int
        The query side's rows, N_q.
    num_key : int
        The key side's rows, N_k.

    Returns
    -------
    BlockMask | torch.Tensor | None
        On the GPU FlexAttention's BlockMask; on the CPU SDPA's dense (N_q, N_k) bool mask, or
        None for a single state.
    """
    if x_off_query.is_cuda:
        return _block_mask(x_off_query, x_off_key, num_query, num_key)
    
    if x_off_query.shape[0] == 2:
        return None
    
    x_state_query, _ = _states_and_ranks(x_off_query, num_query)
    x_state_key, _ = _states_and_ranks(x_off_key, num_key)
    return x_state_query.unsqueeze(1) == x_state_key.unsqueeze(0)


class _Attention(nn.Module):
    """Multi-head attention over packed rows.

    The query and key/value projections, the attention kernel, the output projection.
    FlexAttention, compiled, on the GPU; SDPA on the CPU.

    Parameters
    ----------
    dim_entity : int
        The width of the rows.
    num_heads : int
        The number of attention heads.
    """

    def __init__(self, dim_entity: int, num_heads: int) -> None:
        super().__init__()

        self._projection_query = nn.Linear(dim_entity, dim_entity)
        self._projection_key_value = nn.Linear(dim_entity, 2 * dim_entity)
        self._projection_out = nn.Linear(dim_entity, dim_entity)
        self._num_heads = num_heads

    def forward(
        self, x_query: torch.Tensor, x_key: torch.Tensor, x_mask: BlockMask | torch.Tensor | None
    ) -> torch.Tensor:
        """Have each query row read the key rows that `x_mask` lets it.

        Parameters
        ----------
        x_query : torch.Tensor
            (N_q, D). The query rows.
        x_key : torch.Tensor
            (N_k, D). The key rows, which the values come from too.
        x_mask : BlockMask | torch.Tensor | None
            The key rows each query row reads: FlexAttention's BlockMask, SDPA's dense (N_q, N_k)
            bool mask, or None for every key row.

        Returns
        -------
        torch.Tensor
            (N_q, D). Each query row's read, through the output projection.
        """
        x_q = self._projection_query(x_query)
        x_k, x_v = self._projection_key_value(x_key).chunk(2, dim=1)

        # Both kernels' layout: (1, heads, rows, head dim)
        x_q, x_k, x_v = (
            x_tensor.unflatten(1, (self._num_heads, -1)).transpose(0, 1).unsqueeze(0)
            for x_tensor in (x_q, x_k, x_v)
        )

        # One tensor, (1, H, N_q, D / H): no log-sum-exp is asked for
        if isinstance(x_mask, BlockMask):
            x_read = cast(
                torch.Tensor,
                _flex_attention_compiled(
                    x_q, x_k, x_v, block_mask=x_mask, kernel_options=_FLEX_KERNEL_OPTIONS
                ),
            )
        else:
            x_read = F.scaled_dot_product_attention(x_q, x_k, x_v, attn_mask=x_mask)
        return self._projection_out(x_read.squeeze(0).transpose(0, 1).flatten(1))


class _EntityTransformerBlock(nn.Module):
    def __init__(self, dim_entity: int, dim_feed_forward: int, num_heads: int) -> None:
        super().__init__()

        self._self_attention = _Attention(dim_entity, num_heads)
        self._layer_norm_1 = nn.LayerNorm(dim_entity)
        self._layer_norm_2 = nn.LayerNorm(dim_entity)
        self._mlp = nn.Sequential(
            nn.Linear(dim_entity, dim_feed_forward),
            nn.ReLU(),
            nn.Linear(dim_feed_forward, dim_entity),
        )

    def forward(
        self, x_packed: torch.Tensor, x_mask: BlockMask | torch.Tensor | None
    ) -> torch.Tensor:
        # Pre-LN (norm inside the residual branch): stable at depth, unlike post-LN
        x_normed = self._layer_norm_1(x_packed)
        x_read = self._self_attention(x_normed, x_normed, x_mask)
        x_out = x_packed + x_read

        # Feedforward with residual connection
        x_out = x_out + self._mlp(self._layer_norm_2(x_out))

        return x_out