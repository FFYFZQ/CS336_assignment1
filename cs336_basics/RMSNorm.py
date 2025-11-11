from turtle import forward
import torch
import torch.nn as nn


class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):

        super().__init__()
        self.g = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        in_dyte = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x**2, dim = -1, keepdim = True) + self.eps)

        x_norm = x / rms # pytorch的广播机制，会广播最后一个维度的rms
        output = x_norm * self.g

        return output.to(in_dyte)
        
