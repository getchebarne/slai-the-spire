import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.encoding.map_ import get_encoding_map_dim


_, _, _NUM_CHANNELS = get_encoding_map_dim()


class MapEncoder(nn.Module):
    """
    Lightweight CNN encoder for map state using Global Average Pooling.

    Architecture:
    1. Two convolutional blocks with max pooling
    2. Global Average Pooling (eliminates spatial dimensions)
    3. Single linear projection to embedding_dim
    """

    def __init__(self, kernel_size: int, embedding_dim: int = 128, pad: int = 1):
        super().__init__()

        self._kernel_size = kernel_size
        self._embedding_dim = embedding_dim
        self._pad = pad

        # Reduced channel counts for smaller model
        self._conv_1 = nn.Conv2d(
            in_channels=_NUM_CHANNELS, out_channels=16, kernel_size=kernel_size, padding=pad
        )
        self._conv_2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=kernel_size, padding=pad
        )

        # Pooling layer to downsample the spatial dimensions
        self._max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global average pooling eliminates spatial dimensions entirely
        self._global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Single projection from conv output channels to embedding
        self._projection = nn.Linear(32, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch Conv2d expects (B, C, H, W), but encoding is (B, H, W, C)
        x = x.permute(0, 3, 1, 2)

        # Convolutional block 1
        x = F.relu(self._conv_1(x))
        x = self._max_pool(x)

        # Convolutional block 2
        x = F.relu(self._conv_2(x))
        x = self._max_pool(x)

        # Global average pooling: (B, 32, H', W') -> (B, 32, 1, 1)
        x = self._global_avg_pool(x)

        # Flatten: (B, 32, 1, 1) -> (B, 32)
        x = torch.flatten(x, start_dim=1)

        # Project to embedding dimension
        x = self._projection(x)

        return x
