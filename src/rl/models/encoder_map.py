import torch
import torch.nn as nn
import torch.nn.functional as F


def _calculate_encoder_map_features(
    input_dims: tuple[int, int], kernel_size: int, pad: int = 1
) -> int:
    H, W = input_dims

    # --- Block 1: After _conv_1 and pool ---
    # After _conv_1 (stride=1 is default)
    H = ((H - kernel_size + 2 * pad) // 1) + 1
    W = ((W - kernel_size + 2 * pad) // 1) + 1

    # After self.pool (kernel=2, stride=2)
    H = ((H - 2 + 2 * 0) // 2) + 1
    W = ((W - 2 + 2 * 0) // 2) + 1

    # --- Block 2: After _conv_2 and pool ---
    # After _conv_2
    H = ((H - kernel_size + 2 * pad) // 1) + 1
    W = ((W - kernel_size + 2 * pad) // 1) + 1

    # After self.pool
    H = ((H - 2 + 2 * 0) // 2) + 1
    W = ((W - 2 + 2 * 0) // 2) + 1

    # Final channels from _conv_2 are 64
    final_channels = 64

    # The total number of features is the product of the final dimensions
    return final_channels * H * W


class EncoderMap(nn.Module):
    def __init__(
        self,
        map_height: int,
        map_width: int,
        num_input_channels: int,
        kernel_size: int,
        pad: int = 1,
        embedding_dim: int = 128,
    ):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._kernel_size = kernel_size
        self._embedding_dim = embedding_dim
        self._pad = pad
        self._num_features = _calculate_encoder_map_features(
            (map_height, map_width), kernel_size, pad
        )

        # Convolutional layers
        self._conv_1 = nn.Conv2d(
            in_channels=num_input_channels, out_channels=32, kernel_size=kernel_size, padding=pad
        )
        self._conv_2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=kernel_size, padding=pad
        )

        # Pooling layer to downsample the spatial dimensions
        self._max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self._linear_1 = nn.Linear(self._num_features, 256)
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
