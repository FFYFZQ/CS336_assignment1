import torch
from torch import nn


class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE) module.
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.

        Args:
            theta: float - Θ value for the RoPE
            d_k: int - dimension of query and key vectors
            max_seq_len: int - Maximum sequence length that will be inputted
            device: torch.device | None - Device to store the buffer on
        """
        super().__init__()

        self.device = device
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        frqs = 1.0 / (self.theta ** (torch.arange(0, d_k, 2, device = device) / d_k)) # 化简后的求法
        frqs = frqs.unsqueeze(0) # 变为一个行向量

        token_pos = torch.arange(0, self.max_seq_len, device = device, dtype = torch.float32)
        token_pos = token_pos.unsqueeze(1)

        pre_compute_angle = token_pos @ frqs
        pre_compute_cos = torch.cos(pre_compute_angle)
        pre_compute_sin = torch.sin(pre_compute_angle)

        self.register_buffer("pre_compute_cos", pre_compute_cos, persistent = False)# persistent =false 说明不需要出现在state_dict()中
        self.register_buffer("pre_compute_sin", pre_compute_sin, persistent = False)  # persistent =false 说明不需要出现在state_dict()中

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor and apply RoPE.

        Args:
            x: torch.Tensor - Input tensor of shape (..., seq_len, d_k)
            token_positions: torch.Tensor - Token positions of shape (..., seq_len)

        Returns:
            torch.Tensor - Output tensor of the same shape as x

        Note:
            - x can have an arbitrary number of batch dimensions
            - Use token_positions to slice your precomputed cos and sin tensors
              along the sequence dimension
        """
        cos = self.pre_compute_cos[token_positions] # 使用tensor来索引tensor，取出token_position的值指向位置的pre_compute_cos的值
        # 形状为 (..., seq_len, d_k/2)
        sin = self.pre_compute_sin[token_positions]

        # 由RoPE的定义可知，一个位置对由一个奇数位置和一个偶数位置组成，他们和一个2*2的矩阵相乘得到新的位置的值
        x_odd = x[..., 1::2] # (..., seq_len, d_k/2)
        x_even = x[..., 0::2] # (..., seq_len, d_k/2)

        new_x_odd = x_odd * cos + x_even * sin
        new_x_even = x_odd * (-sin) + x_even * cos

        result_x = torch.empty_like(x)

        result_x[..., 1::2] = new_x_odd
        result_x[..., 0::2] = new_x_even

        return result_x