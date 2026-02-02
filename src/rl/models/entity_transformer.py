import torch
import torch.nn as nn


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

    def forward(
        self,
        x_entity: torch.Tensor,
        x_entity_mask_pad: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process all entities through transformer blocks.

        Args:
            x_entity: Concatenated entity tensor (B, S, D)
            x_entity_mask_pad: Padding mask for entities (B, S), True = padded/invalid

        Returns:
            Transformed entity tensor (B, S, D)
        """
        # Pass all entities through entity transformer
        # Input shape:
        #   `x_entity`: (B, S, D) (float)
        #   `x_entity_mask_pad`: (B, S) (bool)
        for entity_transformer_block in self._entity_transformer_blocks:
            x_entity = entity_transformer_block(x_entity, x_entity_mask_pad)

        # Output shape: (B, S, D)
        return x_entity


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

    def forward(self, x_entity: torch.Tensor, x_mask_pad: torch.Tensor) -> torch.Tensor:
        # Multi-head attention
        x_mha = self._multi_head_attention(
            x_entity, x_entity, x_entity, key_padding_mask=x_mask_pad, need_weights=False
        )[0]
        x_out = self._layer_norm_1(x_entity + x_mha)

        # Feedforward with residual connection
        x_mlp = self._mlp(x_out)
        x_out = self._layer_norm_2(x_out + x_mlp)

        return x_out
