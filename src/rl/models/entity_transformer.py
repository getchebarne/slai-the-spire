import torch
import torch.nn as nn

from src.rl.constants import MAX_SIZE_CARD_PILE_DRAW
from src.rl.types import SliceKind
from src.rl.types import X_Entity


# A batch packs in max(1, rows // this) groups of rows, each group to its own widest row
_ROWS_PER_GROUP = 1024


def _width_groups(
    x_count: torch.Tensor, x_column_key: torch.Tensor, num_groups: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rows in `num_groups` width groups, from their token counts `x_count` (R,): the rows in
    group order (R,), each group's width (G,), and each row's columns in that order, sorted by
    `x_column_key` (R, C)"""

    # The rows sorted by token count, then cut into near-equal groups: a group's width is its
    # last row's count
    x_count, x_order = torch.sort(x_count, stable=True)
    x_widths = torch.stack([x[-1] for x in x_count.tensor_split(num_groups)])
    return x_order, x_widths, torch.argsort(x_column_key[x_order], dim=1, stable=True)


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

    def forward(self, x_packed: torch.Tensor, x_mask_pad: torch.Tensor) -> torch.Tensor:
        # Pre-LN (norm inside the residual branch): stable at depth, unlike post-LN
        x_normed = self._layer_norm_1(x_packed)
        x_mha = self._multi_head_attention(
            x_normed, x_normed, x_normed, key_padding_mask=x_mask_pad, need_weights=False
        )[0]
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
        # has no kind, so it joins the sequence here and leaves it here
        self._entity_global = nn.Parameter(torch.empty(1, 1, dim_entity))
        nn.init.normal_(self._entity_global, std=0.02)

    def forward(
        self, x_entities: X_Entity, x_counts_to_read: torch.Tensor
    ) -> tuple[X_Entity, torch.Tensor, list[int]]:
        """Refine the window together with the global token.

        Returns the refined entities (x (B, S, D), x_slice_kind carried through), the refined
        global token (B, D), and `x_counts_to_read` (k,) as ints, read back to the CPU together
        with the packing widths."""
        batch_size, num_slots, dim = x_entities.x.shape
        num_window_tokens = batch_size * num_slots

        # The sequence as one column of tokens: every row's window slots, then every row's global
        # token. The refined tokens come back in the same layout, so the window and the global
        # tokens leave it as two contiguous blocks
        x_sequence = torch.cat(
            [x_entities.x.flatten(0, 1), self._entity_global.view(1, dim).expand(batch_size, -1)]
        )
        x_valid = torch.cat(
            [
                x_entities.mask,
                torch.ones(batch_size, 1, dtype=torch.bool, device=x_sequence.device),
            ],
            dim=1,
        )
        x_valid_sequence = torch.cat([x_valid[:, :-1].flatten(), x_valid[:, -1]])

        # Pack each group of rows to its true max token count: attention is O(width^2), and a few
        # wide rows would widen a whole batch. The widths' one host sync costs less than attending
        # over a fixed or unpacked width
        num_groups = max(1, batch_size // _ROWS_PER_GROUP)
        if num_groups == 1:
            x_order = torch.arange(batch_size, device=x_sequence.device)
            x_widths = x_valid.sum(dim=1).max().unsqueeze(0)
            x_column = torch.argsort(~x_valid, dim=1, stable=True, descending=False)
        else:
            x_order, x_widths, x_column = _width_groups(x_valid.sum(dim=1), ~x_valid, num_groups)

        # The widths and the counts to read, in one read back to the CPU
        widths_and_counts = torch.cat([x_widths, x_counts_to_read]).tolist()
        pack_widths = widths_and_counts[:num_groups]
        counts = widths_and_counts[num_groups:]

        # Each row's columns in x_order, valid first (its window slots, then its global token, the
        # last column), then its pads, as indices into the sequence; a group keeps its rows' first
        # pack_width
        x_row = x_order.unsqueeze(1)
        x_slot = torch.where(
            x_column < num_slots, x_row * num_slots + x_column, num_window_tokens + x_row
        )
        x_keep_groups = [
            x_slot_group[:, :pack_width]
            for x_slot_group, pack_width in zip(x_slot.tensor_split(num_groups), pack_widths)
        ]

        # One gather packs every group and one scatter puts them back. The blocks get copies, not
        # views: a view's base would compile them again
        x_keep = torch.cat([x_keep_group.flatten() for x_keep_group in x_keep_groups])
        if num_groups == 1:
            x_refined = self._refine(
                x_sequence[x_keep_groups[0]],
                ~x_valid_sequence[x_keep_groups[0]],  # MHA wants True = padded
            ).flatten(0, 1)
        else:
            x_packed_groups = x_sequence.index_select(0, x_keep).split(
                [x_keep_group.numel() for x_keep_group in x_keep_groups]
            )
            x_refined = torch.cat(
                [
                    self._refine(
                        x_packed.unflatten(0, x_keep_group.shape).clone(),
                        ~x_valid_sequence[x_keep_group],
                    ).flatten(0, 1)
                    for x_keep_group, x_packed in zip(x_keep_groups, x_packed_groups)
                ]
            )

        # Scatter the refined tokens back to their original slots. Zeros, not empty_like:
        # downstream masks multiply (x*0) and NaN*0 = NaN, so uninitialized bytes would propagate
        x_out = torch.zeros_like(x_sequence)
        x_out.index_copy_(0, x_keep, x_refined)
        x_out_window, x_out_global = x_out.split([num_window_tokens, batch_size])
        return (
            X_Entity(x_out_window.view(batch_size, num_slots, dim), x_entities.x_slice_kind),
            x_out_global,
            counts,
        )

    def _refine(self, x_packed: torch.Tensor, x_mask_pad: torch.Tensor) -> torch.Tensor:
        """The blocks, then the final norm, over packed rows"""
        for entity_transformer_block in self._entity_transformer_blocks:
            x_packed = entity_transformer_block(x_packed, x_mask_pad)
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
        x_mask_pad: torch.Tensor,
        x_global_token: torch.Tensor,
        x_depth: torch.Tensor,
        x_mask_pile_pad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Over packed rows whose first columns hold the draw pile (`x_depth`'s width): every
        token reads the pile tagged with its depths, then each row's global token (R, D) reads
        the order-aware tokens. Pre-LN and residual, then one final norm over both outputs, as
        in the transformer"""
        # The pile's tokens, tagged with their depths
        x_pile = self._layer_norm_1(
            x_packed[:, : x_depth.shape[1]] + self._draw_depth_emb(x_depth)
        )

        # Every token reads the pile, then the feedforward
        x_read = self._cross_attention(
            self._layer_norm_1(x_packed),
            x_pile,
            x_pile,
            key_padding_mask=x_mask_pile_pad,
            need_weights=False,
        )[0]
        x_out = x_packed + x_read
        x_out = x_out + self._mlp(self._layer_norm_2(x_out))

        # The global token reads the order-aware tokens
        x_normed = self._layer_norm_3(x_out)
        x_read_global = self._readout_attention(
            self._layer_norm_3(x_global_token).unsqueeze(1),
            x_normed,
            x_normed,
            key_padding_mask=x_mask_pad,
            need_weights=False,
        )[0]
        return self._norm_final(x_out), self._norm_final(x_global_token + x_read_global.squeeze(1))


class DrawOrderLayer(nn.Module):
    def __init__(self, dim_entity: int, dim_feed_forward: int, num_heads: int) -> None:
        super().__init__()

        self._draw_order_block = _DrawOrderBlock(dim_entity, dim_feed_forward, num_heads)

    def forward(
        self,
        x_entities: X_Entity,
        x_global_token: torch.Tensor,
        x_depth: torch.Tensor,
        x_rows: torch.Tensor,
    ) -> tuple[X_Entity, torch.Tensor]:
        """The layer over the states `x_rows`, each with an ordered draw pile, of the
        transformer's refined window (B, S, D) and global tokens (B, D), given the window's draw
        depths (B, S).

        Returns those states' order-aware window (x (R, S, D), x_slice_kind carried through) and
        global tokens (R, D)."""
        num_rows = x_rows.shape[0]
        num_slots, dim = x_entities.x.shape[1:]
        x_slice_kind = x_entities.x_slice_kind[x_rows]
        x_valid = x_slice_kind >= 0
        x_is_draw = x_slice_kind == SliceKind.CARD_PILE_DRAW

        # Width groups, as the transformer's: the rows sorted by token count and cut into
        # near-equal groups, each packed to its last row's count and its pile to its largest
        num_groups = max(1, num_rows // _ROWS_PER_GROUP)
        x_order, x_widths, x_column = _width_groups(
            x_valid.sum(dim=1), (~x_is_draw).long() + (~x_valid).long(), num_groups
        )
        x_count_pile = x_is_draw.sum(dim=1)[x_order]
        pack_widths, pile_widths = torch.stack(
            [x_widths, torch.stack([x.max() for x in x_count_pile.tensor_split(num_groups)])]
        ).tolist()

        # Each row's columns in x_order: its draw tokens first, so the pile is a prefix, then its
        # other valid tokens, then its pads, as indices into these rows' window
        x_slot = x_order.unsqueeze(1) * num_slots + x_column
        x_keep_groups = [
            x_slot_group[:, :pack_width]
            for x_slot_group, pack_width in zip(x_slot.tensor_split(num_groups), pack_widths)
        ]
        x_keep = torch.cat([x_keep_group.flatten() for x_keep_group in x_keep_groups])

        # One gather packs every group from the batch's window, where a kept slot's row is its
        # state's. The block gets the packed tokens as copies, not views, as the transformer's do
        x_keep_batch = x_rows[x_keep // num_slots] * num_slots + x_keep % num_slots
        x_packed_groups = (
            x_entities.x.flatten(0, 1)
            .index_select(0, x_keep_batch)
            .split([x_keep_group.numel() for x_keep_group in x_keep_groups])
        )
        x_valid_flat, x_is_draw_flat = x_valid.flatten(), x_is_draw.flatten()
        x_depth_flat = x_depth[x_rows].flatten()
        x_refined, x_refined_global = zip(
            *[
                self._draw_order_block(
                    x_packed.unflatten(0, x_keep_group.shape).clone(),
                    ~x_valid_flat[x_keep_group],  # MHA wants True = padded
                    x_global_group,
                    x_depth_flat[x_keep_group[:, :pile_width]],
                    ~x_is_draw_flat[x_keep_group[:, :pile_width]],
                )
                for x_keep_group, x_packed, x_global_group, pile_width in zip(
                    x_keep_groups,
                    x_packed_groups,
                    x_global_token[x_rows[x_order]].tensor_split(num_groups),
                    pile_widths,
                )
            ]
        )

        # Scatter the refined tokens back to their slots, zeros elsewhere, and the global tokens
        # back to row order
        x_out = x_entities.x.new_zeros(num_rows * num_slots, dim)
        x_out.index_copy_(0, x_keep, torch.cat([x.flatten(0, 1) for x in x_refined]))
        x_out_global = torch.cat(x_refined_global)
        return (
            X_Entity(x_out.view(num_rows, num_slots, dim), x_slice_kind),
            torch.empty_like(x_out_global).index_copy_(0, x_order, x_out_global),
        )
