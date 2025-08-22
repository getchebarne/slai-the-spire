import torch
import torch.nn as nn


class TransformerEntity(nn.Module):
    def __init__(self, dim_embedding: int, dim_mlp: int, num_heads: int):
        super().__init__()

        self._multi_head_attention = nn.MultiheadAttention(
            dim_embedding, num_heads, batch_first=True
        )
        self._layer_norm_1 = nn.LayerNorm(dim_embedding)
        self._layer_norm_2 = nn.LayerNorm(dim_embedding)
        self._mlp = nn.Sequential(
            nn.Linear(dim_embedding, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, dim_embedding),
        )

    def forward(self, x_entity: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        # Multi-head attention
        x_mha = self._mha_with_fallback(x_entity, x_mask)
        x_out = self._layer_norm_1(x_entity + x_mha)

        # Feedforward with residual connection
        x_mlp = self._mlp(x_out)
        x_out = self._layer_norm_2(x_out + x_mlp)

        return x_out

    def _mha_with_fallback(self, x_entity: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        # x_entity: (batch_size, num_entities, dim_entity)
        # x_mask: (batch_size, num_entities)
        x_empty = torch.all(x_mask, dim=1)  # (batch_size,)

        if torch.all(x_empty):
            # Batch full of null entities. Return zeroed out tensor
            return torch.zeros_like(x_entity)

        if not torch.any(x_empty):
            # No null entities in batch. Should be fairly common
            return self._multi_head_attention(
                x_entity, x_entity, x_entity, key_padding_mask=x_mask, need_weights=False
            )[0]

        # Mix of empty and non-empty hands
        x_non_empty = ~x_empty
        x_entity_non_empty = x_entity[x_non_empty]
        x_mask_non_empty = x_mask[x_non_empty]

        # Process non-empty hands
        x_out_non_empty, _ = self._multi_head_attention(
            x_entity_non_empty,
            x_entity_non_empty,
            x_entity_non_empty,
            key_padding_mask=x_mask_non_empty,
            need_weights=False,
        )

        # Create all-zeros output tensor and fill non-empty hands w/ MHA output
        x_out = torch.zeros_like(x_entity)
        x_out[x_non_empty] = x_out_non_empty  # (batch_size, num_entities, dim_embedding)

        return x_out
