import torch
from torch import nn
from jaxtyping import Float
import math
from torch import Tensor
import torch.nn as nn


class SwiGLU_FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dtype=None, device=None):

        super().__init__()
        self.dtype = dtype
        self.device = device
        self.d_model = d_model
        # d_ff = 8 / 3 * d_model
        self.d_ff = d_ff
        self.gluw1 = nn.Parameter(
            torch.empty(self.d_ff, self.d_model, dtype=self.dtype, device=self.device)
        )
        self.w2 = nn.Parameter(
            torch.empty(self.d_model, self.d_ff, dtype=self.dtype, device=self.device)
        )
        self.gluw3 = nn.Parameter(
            torch.empty(self.d_ff, self.d_model, dtype=self.dtype, device=self.device)
        )
        self.weight_init()  # 初始化权重参数

    def weight_init(self):
        std = math.sqrt(2 / (self.d_ff + self.d_model))
        torch.nn.init.trunc_normal_(
            self.gluw1, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std
        )
        torch.nn.init.trunc_normal_(
            self.w2, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std
        )
        torch.nn.init.trunc_normal_(
            self.gluw3, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std
        )

    def silu(self, input: Float[Tensor, "..."]):
        return input * torch.sigmoid(input) #silu是逐元素乘法

    def swiglu(self, input: Float[Tensor, "... d_model"]) -> Float[Tensor, "d_model d_ff"]:
        silu_output = self.silu(input @ self.gluw1.T)
        output = silu_output * (input @ self.gluw3.T)
        return output

    def forward(self, x: Float[Tensor, "... d_model"]):
        swiglu_output = self.swiglu(x)
        output = swiglu_output @ self.w2.T

        return output
