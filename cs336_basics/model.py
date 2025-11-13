import torch
from torch import nn
from .transformer import TransformerBlock
from .embedding_layer import Embedding
from .RMSNorm import RMSNorm
from .linear_layer import Linear


class Transformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        vocab_size: int,
        context_length: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model

        # Token embedding layer
        self.token_embedding = Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, **factory_kwargs
        )

        # Stack of Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_heads=num_heads,
                    rope_theta=rope_theta,
                    max_seq_len = context_length,
                    **factory_kwargs,
                )
                for _ in range(num_layers)
            ]
        )

        # Final RMSNorm layer
        self.final_norm = RMSNorm(d_model=d_model, **factory_kwargs)

        # Output projection using custom Linear layer (no bias)
        self.lm_head = Linear(
            in_features=d_model, out_features=vocab_size, **factory_kwargs
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer LM.

        Args:
            input_ids: Input token indices of shape (batch_size, seq_len)

        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
        """
        # Token embedding: (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x = self.token_embedding(input_ids)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Final normalization
        x = self.final_norm(x)

        # Project to vocabulary size for language modeling
        logits = self.lm_head(x)

        return logits
