from typing import Any

import torch
import torch.nn.functional as F


# The copies of a table that SpreadEmbedding's backward sums over
_SPREAD_COPIES = 256


class SpreadEmbedding(torch.autograd.Function):
    """F.embedding, whose backward sums each table row's gradient over copies of the table.

    There are `_SPREAD_COPIES` copies: the lookup at flat position `i` adds into copy
    `i % _SPREAD_COPIES`, then the copies add up. The same sums in another order, in short loops.
    """

    @staticmethod
    def forward(ctx: Any, x_weight: torch.Tensor, x_idx_emb: torch.Tensor) -> torch.Tensor:
        """Look up the rows of `x_weight` at `x_idx_emb`, as F.embedding does.

        Parameters
        ----------
        x_weight : torch.Tensor
            (R, D). The table: R rows of width D.
        x_idx_emb : torch.Tensor
            (...) long. The rows to look up, in any shape.

        Returns
        -------
        torch.Tensor
            (..., D). Each lookup's row.
        """
        ctx.save_for_backward(x_idx_emb)
        ctx.num_embeddings = x_weight.shape[0]

        return F.embedding(x_idx_emb, x_weight)

    @staticmethod
    def backward(ctx: Any, x_grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Sum each table row's gradient over `_SPREAD_COPIES` copies of the table.

        Parameters
        ----------
        x_grad : torch.Tensor
            (..., D). Each lookup's gradient.

        Returns
        -------
        x_weight_grad : torch.Tensor
            (R, D). The table's gradient.
        None
            The indexes take no gradient.
        """
        (x_idx_emb,) = ctx.saved_tensors  # (...) long

        # The first row of each lookup's copy in the stacked table:
        # Lookup `i` (flat position) goes to copy `i % _SPREAD_COPIES`,
        # whose rows start at `(i % _SPREAD_COPIES) * R`
        x_start = (
            torch.arange(x_idx_emb.numel(), device=x_idx_emb.device).view_as(
                x_idx_emb
            )
            % _SPREAD_COPIES
        ) * ctx.num_embeddings  # (...)

        # F.embedding's own backward: row `r` of the result sums the gradients of the lookups at
        # row `r`, as `torch.zeros(num_weights, D).index_add_(0, indices, grad_output)` would
        x_grad_copy = torch.ops.aten.embedding_dense_backward(
            x_grad,  # (..., D)
            x_start + x_idx_emb,  # (...): rows of the stacked table
            ctx.num_embeddings * _SPREAD_COPIES,
            padding_idx=-1,
            scale_grad_by_freq=False,
        )  # (_SPREAD_COPIES * R, D)
        x_weight_grad = x_grad_copy.view(_SPREAD_COPIES, ctx.num_embeddings, -1).sum(0)  # (R, D)

        return x_weight_grad, None
