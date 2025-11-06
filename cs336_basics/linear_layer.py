import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class Linear(nn.Module):
    """
    Linear transformation module without bias.

    Performs: output = input @ W.T
    where W has shape (out_features, in_features)
    """

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        Initialize the Linear module.

        Args:
            in_features: Final dimension of the input
            out_features: Final dimension of the output
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()

        # TODO: Create weight parameter W with shape (out_features, in_features)
        # Use nn.Parameter to make it a learnable parameter
        # Hint: torch.empty(out_features, in_features, device=device, dtype=dtype)
        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        # TODO: Initialize weights using torch.nn.init.trunc_normal_
        # Standard initialization for transformers
        torch.nn.init.trunc_normal_(self.W, mean=0.0, std=0.02)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        """
        Apply linear transformation to input.

        Args:
            x: Input tensor with last dimension = in_features

        Returns:
            Output tensor with last dimension = out_features
        """
        # TODO: Implement the linear transformation
        # Hint: Use torch.matmul or @ operator
        # Remember: output = x @ W.T
        return x @ self.W.T
