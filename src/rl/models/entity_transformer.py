from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import flex_attention

from src.rl.types import Jagged


# FlexAttention's block: on the GPU, the sequence is padded to a multiple of it
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

        # Each slot's state. The padded slots past the last state form state `batch_size`
        x_slot_state = torch.full((num_slots,), batch_size, dtype=torch.long, device=x_rows.device)
        x_slot_state[x_place_row] = x_state
        x_slot_state[x_place_global] = torch.arange(batch_size, device=x_rows.device)

        # Run Attention
        x_refined = self._refine(x_sequence, _same_state_mask(x_slot_state, x_off))

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


def _block_mask(x_slot_state: torch.Tensor, x_off: torch.Tensor) -> BlockMask:
    """Build FlexAttention's same-state mask on the device.

    Each slot attends the slots of its own state.

    Parameters
    ----------
    x_slot_state : torch.Tensor
        (num_slots,) long. Each slot's state, num_slots a multiple of _FLEX_BLOCK; the padded
        slots form state B.
    x_off : torch.Tensor
        (B + 1,) long. Each state's first slot, then the end.

    Returns
    -------
    BlockMask
        The same-state mask.
    """
    num_slots = x_slot_state.shape[0]
    num_blocks = num_slots // _FLEX_BLOCK

    # Every state's first slot, the padded slots' state too, then the sequence's end
    x_starts = F.pad(x_off, (0, 1), value=num_slots)  # (B + 2,)

    # Each slot's state, one row per block
    x_slot_state_by_block = x_slot_state.view(num_blocks, _FLEX_BLOCK)  # (num_blocks, _FLEX_BLOCK)

    # The state at each block's first slot, and at its last slot
    x_state_first = x_slot_state_by_block[:, 0]  # (num_blocks,)
    x_state_last = x_slot_state_by_block[:, -1]  # (num_blocks,)

    # Where those states start and end: the first and last slot the block's slots attend
    x_slot_first = x_starts[x_state_first]  # (num_blocks,)
    x_slot_last = x_starts[x_state_last + 1] - 1  # (num_blocks,)

    # The blocks those two slots are in: the first and last key block the query block attends
    x_block_first = x_slot_first // _FLEX_BLOCK  # (num_blocks,)
    x_block_last = x_slot_last // _FLEX_BLOCK  # (num_blocks,)

    # `from_kv_blocks` wants each query block's key blocks as wide as the key blocks: its first
    # (x_block_last - x_block_first + 1) count
    x_rank = torch.arange(num_blocks, device=x_block_first.device)  # (num_blocks,)
    x_blocks_key = x_block_first.unsqueeze(1) + x_rank  # (num_blocks, num_blocks)

    def same_state(
        x_b: torch.Tensor, x_h: torch.Tensor, x_q_idx: torch.Tensor, x_kv_idx: torch.Tensor
    ) -> torch.Tensor:
        return x_slot_state[x_q_idx] == x_slot_state[x_kv_idx]

    return BlockMask.from_kv_blocks(
        (x_block_last - x_block_first + 1).int().view(1, 1, -1),
        x_blocks_key.clamp(max=num_blocks - 1).int().view(1, 1, -1, num_blocks),
        BLOCK_SIZE=_FLEX_BLOCK,
        mask_mod=same_state,
        seq_lengths=(num_slots, num_slots),
    )


def _same_state_mask(
    x_slot_state: torch.Tensor, x_off: torch.Tensor
) -> BlockMask | torch.Tensor | None:
    """Build the mask that lets each slot attend the slots of its own state.

    Parameters
    ----------
    x_slot_state : torch.Tensor
        (num_slots,) long. Each slot's state; the padded slots form state B.
    x_off : torch.Tensor
        (B + 1,) long. Each state's first slot, then the end.

    Returns
    -------
    BlockMask | torch.Tensor | None
        On the GPU FlexAttention's BlockMask; on the CPU SDPA's dense (num_slots, num_slots) bool
        mask, or None for a single state.
    """
    if x_slot_state.is_cuda:
        return _block_mask(x_slot_state, x_off)

    if x_off.shape[0] == 2:
        return None

    return x_slot_state.unsqueeze(1) == x_slot_state.unsqueeze(0)


class _Attention(nn.Module):
    """Multi-head self-attention over packed rows.

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
        self, x_packed: torch.Tensor, x_mask: BlockMask | torch.Tensor | None
    ) -> torch.Tensor:
        """Have each row read the rows that `x_mask` lets it.

        Parameters
        ----------
        x_packed : torch.Tensor
            (N, D). The packed rows.
        x_mask : BlockMask | torch.Tensor | None
            The rows each row reads: FlexAttention's BlockMask, SDPA's dense (N, N) bool mask, or
            None for every row.

        Returns
        -------
        torch.Tensor
            (N, D). Each row's read, through the output projection.
        """
        x_q = self._projection_query(x_packed)
        x_k, x_v = self._projection_key_value(x_packed).chunk(2, dim=1)

        # Both kernels' layout: (1, heads, rows, head dim)
        x_q, x_k, x_v = (
            x_tensor.unflatten(1, (self._num_heads, -1)).transpose(0, 1).unsqueeze(0)
            for x_tensor in (x_q, x_k, x_v)
        )

        # One tensor, (1, H, N, D / H): no log-sum-exp is asked for
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
        x_read = self._self_attention(x_normed, x_mask)
        x_out = x_packed + x_read

        # Feedforward with residual connection
        x_out = x_out + self._mlp(self._layer_norm_2(x_out))

        return x_out