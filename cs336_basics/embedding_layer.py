import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
import math


class Embedding(nn.Module):
    """
    Embedding lookup module.

    Performs embedding lookup for given token IDs.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, device=None, dtype=None
    ):
        """
        Initialize the Embedding module.

        Args:
            num_embeddings: Size of the vocabulary
            embedding_dim: Dimension of the embedding vectors (d_model)
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        self.weights = nn.Parameter(torch.empty([num_embeddings, embedding_dim], device = device, dtype = dtype))
        torch.nn.init.trunc_normal_(self.weights, a = -3.0, b = 3.0)



    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        """
        Lookup the embedding vectors for the given token IDs.

        Args:
            token_ids: Tensor of token IDs to look up embeddings for

        Returns:
            Embedding vectors for the given token IDs
        """
        # 如果传入了一个向量作为索引，那么就会进行 “行选择” 的操作，针对每一个int选取对应的行
        return self.weights[token_ids]

