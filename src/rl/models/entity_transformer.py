import torch
import torch.nn as nn

from src.rl.types import TPadded


class EntityTransformer(nn.Module):
    def __init__(self, dim_embedding: int, dim_feed_forward: int, num_heads: int, num_blocks: int):
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

    def forward(self, t_padded: TPadded) -> TPadded:
        """Process all entities through transformer blocks.

        Args:
            t_padded: TPadded (x: (B, S, D), mask: (B, S) True = valid). The valid-mask is flipped
                to a key-padding mask (True = padded) for attention.

        Returns:
            Transformed entities as a TPadded (x (B, S, D) refined, mask carried through unchanged)
        """
        t_entity = t_padded.x
        t_entity_mask_pad = ~t_padded.mask  # TPadded mask is True=valid; MHA wants True=padded
        for entity_transformer_block in self._entity_transformer_blocks:
            t_entity = entity_transformer_block(t_entity, t_entity_mask_pad)

        return TPadded(t_entity, t_padded.mask)


class _EntityTransformerBlock(nn.Module):
    def __init__(self, dim_embedding: int, dim_feed_forward: int, num_heads: int):
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

    def forward(self, t_entity: torch.Tensor, t_mask_pad: torch.Tensor) -> torch.Tensor:
        # Multi-head attention
        t_mha = self._multi_head_attention(
            t_entity, t_entity, t_entity, key_padding_mask=t_mask_pad, need_weights=False
        )[0]
        t_out = self._layer_norm_1(t_entity + t_mha)

        # Feedforward with residual connection
        t_mlp = self._mlp(t_out)
        t_out = self._layer_norm_2(t_out + t_mlp)

        return t_out
