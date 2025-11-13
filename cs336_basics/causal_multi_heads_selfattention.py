from .utils import scaled_dot_product_attention
import torch
from torch import nn
from .linear_layer import Linear
from .rope import RoPE
from einops import rearrange


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_sqe_len: int = 2048, theta: float = 0.0, device = None, dtype = None):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head= self.d_model // self.num_heads # 整除
        # 创建投影矩阵
        self.W_qkv = Linear(d_model, 3 * d_model, device, dtype) # 合并qkv矩阵，把原来需要三个乘法的转化成一个
        self.W_out = Linear(d_model, d_model, device, dtype)
        if theta:
            self.rope = RoPE(theta, self.d_head, max_sqe_len)
        else:
            self.rope = None



    def forward(self, x, token_position:torch.Tensor = None):
        batch_size, seq_len, _ = x.shape
        qkv = self.W_qkv(x) # 跟随x创建在和x相同的device上
        
        q, k, v = rearrange(qkv, 'batch seq (three num_heads d_head) -> three batch num_heads seq d_head', three = 3, num_heads = self.num_heads)
        
        if self.rope is not None:
            if token_position is None:
                token_position = torch.arange(seq_len, device = x.device)
            q = self.rope(q, token_position)
            k = self.rope(k, token_position)
        
        # 使用trill创建一个下三角的矩阵
        mask = torch.tril(torch.ones(seq_len, seq_len, device = x.device))
        mask = mask.unsqueeze(0).unsqueeze(0)

        # 应用 scaled dot production
        attn_output = scaled_dot_product_attention(q, k, v, mask = mask)

        attn_output = rearrange(attn_output, 'batch heads seq d_head -> batch seq (heads d_head)')

        output = self.W_out(attn_output)

        return output