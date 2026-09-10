import torch
import torch.nn as nn

from src.rl.types import X_Entity


class EntityTransformer(nn.Module):
    def __init__(
        self, dim_embedding: int, dim_feed_forward: int, num_heads: int, num_blocks: int
    ) -> None:
        super().__init__()

        self._dim_embedding = dim_embedding
        self._dim_feed_forward = dim_feed_forward
        self._num_heads = num_heads
        self._num_blocks = num_blocks

        self._entity_transformer_blocks = nn.ModuleList(
            [
                _EntityTransformerBlock(dim_embedding, dim_feed_forward, num_heads)
                for _ in range(num_blocks)
            ]
        )

        # Pre-LN blocks leave the residual stream unnormalized; one final norm
        self._norm_final = nn.LayerNorm(dim_embedding)

        # Learned global token, refined with the entities (replaces pooling). It
        # has no kind, so it joins the sequence here and leaves it here
        self._entity_global = nn.Parameter(torch.empty(1, 1, dim_embedding))
        nn.init.normal_(self._entity_global, std=0.02)

    def forward(self, x_entities: X_Entity) -> tuple[X_Entity, torch.Tensor]:
        """Refine the window together with the global token.

        Returns the refined entities (x (B, S, D), idxs carried through; pads exact
        zeros) and the refined global token (B, D)."""
        batch_size, _, dim = x_entities.x.shape
        x_x = torch.cat([x_entities.x, self._entity_global.expand(batch_size, -1, -1)], dim=1)
        x_valid = torch.cat(
            [x_entities.mask, torch.ones(batch_size, 1, dtype=torch.bool, device=x_x.device)],
            dim=1,
        )

        # Pack to the batch's true max token count: attention is O(width^2), and CPU-eager has no
        # recompilation penalty for a per-batch width. The int(max) is a per-forward host sync,
        # but it measured ~free on GPU (2026-07 probe) — and the alternatives lost: no-pack
        # ballooned the update (199-token attention, 12.0 s/iter) and a fixed width paid ~2.3x
        # the transformer for widths that are typically 20-31 tokens
        pack_width = int(x_valid.sum(dim=1).max())
        x_keep = torch.argsort(~x_valid, dim=1, stable=True, descending=False)[:, :pack_width]
        x_keep_x = x_keep.unsqueeze(-1).expand(-1, -1, dim)
        x_packed = x_x.gather(1, x_keep_x)
        x_mask_pad = ~x_valid.gather(1, x_keep)  # MHA wants True = padded
        for entity_transformer_block in self._entity_transformer_blocks:
            x_packed = entity_transformer_block(x_packed, x_mask_pad)
        x_packed = self._norm_final(x_packed)

        # Scatter the refined tokens back to their original slots. Padding must be zeroed (not
        # empty_like): downstream masks multiply (x*0), and NaN*0=NaN, so uninitialized NaN-pattern
        # bytes propagate — observed crashing the x86 CPU path with NaN-pattern garbage
        x_out = torch.zeros_like(x_x)
        x_out.scatter_(1, x_keep_x, x_packed)
        return X_Entity(x_out[:, :-1], x_entities.x_slice_kind), x_out[:, -1]


class _EntityTransformerBlock(nn.Module):
    def __init__(self, dim_embedding: int, dim_feed_forward: int, num_heads: int) -> None:
        super().__init__()

        self._dim_embedding = dim_embedding
        self._dim_feed_forward = dim_feed_forward
        self._num_heads = num_heads

        self._multi_head_attention = nn.MultiheadAttention(
            dim_embedding, num_heads, batch_first=True
        )
        self._layer_norm_1 = nn.LayerNorm(dim_embedding)
        self._layer_norm_2 = nn.LayerNorm(dim_embedding)
        self._mlp = nn.Sequential(
            nn.Linear(dim_embedding, dim_feed_forward),
            nn.ReLU(),
            nn.Linear(dim_feed_forward, dim_embedding),
        )

    def forward(self, x_entity: torch.Tensor, x_mask_pad: torch.Tensor) -> torch.Tensor:
        # Pre-LN (norm inside the residual branch): stable at depth, unlike the
        # post-LN form this replaced for EFFECT-III's 3-block stack
        x_normed = self._layer_norm_1(x_entity)
        x_mha = self._multi_head_attention(
            x_normed, x_normed, x_normed, key_padding_mask=x_mask_pad, need_weights=False
        )[0]
        x_out = x_entity + x_mha

        # Feedforward with residual connection
        x_out = x_out + self._mlp(self._layer_norm_2(x_out))

        return x_out
