import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.encoding.map_ import get_encoding_map_dim


_, _, _NUM_CHANNELS = get_encoding_map_dim()


class MapEncoder(nn.Module):
    def __init__(self, kernel_size: int, embedding_dim: int = 128, pad: int = 1):
        super().__init__()

        self._kernel_size = kernel_size
        self._embedding_dim = embedding_dim
        self._pad = pad

        # Convolutional layers
        self._conv_1 = nn.Conv2d(
            in_channels=_NUM_CHANNELS, out_channels=32, kernel_size=kernel_size, padding=pad
        )
        self._conv_2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=kernel_size, padding=pad
        )

        # Pooling layer to downsample the spatial dimensions
        self._max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self._linear_1 = nn.LazyLinear(256)
        self._linear_2 = nn.Linear(256, embedding_dim)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional block 1
        x = F.relu(self._conv_1(x))
        x = self._max_pool(x)

        # Convolutional block 2
        x = F.relu(self._conv_2(x))
        x = self._max_pool(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch Conv2d expects (B, C, H, W), but your encoding is (H, W, C).
        # We permute the dimensions to match the expected format.
        x = x.permute(0, 3, 1, 2)

        # Pass through convolutional layers
        x = self._forward_conv(x)

        # Flatten the output for the fully connected layers
        # (B, C_out, H_out, W_out) -> (B, C_out * H_out * W_out)
        x = torch.flatten(x, start_dim=1)

        # Pass through fully connected layers
        x = self._linear_1(x)
        x = F.relu(x)
        x = self._linear_2(x)

        return x
