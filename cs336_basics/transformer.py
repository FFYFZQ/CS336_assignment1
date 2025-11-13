import torch
from torch import nn
from .causal_multi_heads_selfattention import CausalMultiHeadSelfAttention
from .feedforward import SwiGLU_FFN
from .RMSNorm import RMSNorm
 
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int = 2048, rope_theta: float = 0.0,  dtype = None, device = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rms1 = RMSNorm(d_model, device = device, dtype = dtype) # transformer层之前
        self.rms2 = RMSNorm(d_model, device = device, dtype = dtype) # ffn层之前
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, max_seq_len, rope_theta, device, dtype)
        self.ffn = SwiGLU_FFN(d_model, d_ff, dtype, device)

    def forward(self, input: torch.Tensor, token_position = None):
        attn_output = self.attn(self.rms1(input), token_position)
        layer_output = attn_output + input
        output = self.ffn(self.rms2(layer_output)) + layer_output

        return output



