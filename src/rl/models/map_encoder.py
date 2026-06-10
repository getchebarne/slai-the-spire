import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.encoding.map_ import _NUM_CHANNELS


class MapEncoder(nn.Module):
    """
    Lightweight CNN encoder for the map grid, preserving per-column identity.

    The convolutions mix locally, pooling collapses height only, so each map
    column keeps its own embedding (the map head scores columns individually).
    Output: (B, MAP_WIDTH, embedding_dim). Width preservation assumes the convs
    are shape-preserving (kernel_size=3 with pad=1, the configured values).
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

        # Downsample height only — pooling width would merge columns
        self._max_pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        # Per-column projection from conv channels to embedding
        self._projection = nn.Linear(32, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Envs sit on one map node for many steps, so batches are grid-redundant
        # (~2x in update minibatches): encode unique grids once and gather back —
        # bitwise exact. The convs run as unfold+GEMM (same weights, same math),
        # dodging the slow CPU convolution_backward path at these tiny shapes;
        # measured together: 1.88x on the encoder, 1.20x on the full update step.
        b = x.shape[0]
        uniq, inverse = torch.unique(x.reshape(b, -1), dim=0, return_inverse=True)
        # Conv2d expects (U, C, H, W), but the encoding is (H, W, C)
        u = uniq.reshape(-1, *x.shape[1:]).permute(0, 3, 1, 2)

        # Convolutional block 1
        u = F.relu(self._conv_gemm(u, self._conv_1))
        u = self._max_pool(u)

        # Convolutional block 2
        u = F.relu(self._conv_gemm(u, self._conv_2))
        u = self._max_pool(u)

        # Collapse the remaining height: (U, 32, H', W) -> (U, W, 32)
        u = u.mean(dim=2).permute(0, 2, 1)

        # Project each column to the embedding dimension, expand back to (B, W, E)
        return self._projection(u)[inverse]

    @staticmethod
    def _conv_gemm(x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
        b, _, h, w = x.shape
        cols = F.unfold(x, kernel_size=conv.kernel_size, padding=conv.padding)
        out = conv.weight.reshape(conv.out_channels, -1) @ cols
        return (out + conv.bias.view(1, -1, 1)).view(b, conv.out_channels, h, w)
