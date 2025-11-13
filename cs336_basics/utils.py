"""
实现了函数结构以及训练函数时的一些基础函数
"""

import torch
import math
from jaxtyping import Float, Int
from torch import Tensor
from torch import nn
from collections.abc import Iterable

"""
 实现了一个scaled_dot_product_attention，并且在计算每一个softmax时减去一个最大项
 来保持数值稳定性（softmax操作的所有指数同时减去一个数，是不影响softmax的结果的）
"""


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    d_k = query.shape[-1]
    d_v = value.shape[-1]  # 取两者的最后一维的维度

    scores = query @ key.transpose(-1, -2)
    if mask is not None:
        scores.masked_fill_(
            mask == False, -1e9
        )  # masked_fill_是一个原地操作，这个意思是在mask的值等于false的时候，替换值为-1e9
    scores /= math.sqrt(d_k)

    attention = softmax(scores, dim=-1)

    output = attention @ value

    return output


# 保持数值稳定性的softmax操作
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:

    if dim < 0:
        dim += len(x.shape)
    x_max = x.max(dim=dim, keepdim=True)[0]
    # 使用数值稳定性技巧，同时减去最大值
    x = x - x_max

    x = torch.exp(x)
    x_sum = x.sum(dim=dim, keepdim=True)

    output = x / x_sum
    return output


# 计算cross_entropy,注意数值稳定
def cross_entropy(
    logits: Float[Tensor, "batch_size vocab_size"],
    position: Int[Tensor, "batch_size"],
):
    logits_max = torch.max(
        logits, dim=-1, keepdim=True
    ).values  # torch.max返回一个命名元组，包含values和indices两个属性，我们需要values
    logits -= logits_max

    loss = -logits.gather(dim=-1, index=position.unsqueeze(-1)).squeeze(-1) + torch.log(
        torch.sum(torch.exp(logits), dim=-1)
    )

    return loss.mean()  # 返回平均损失


def cosine_annealing(t, lr_max, lr_min, t_w, t_c):
    lr = 0
    if t < t_w:
        lr = t / t_w * lr_max
    elif t > t_c:
        lr = lr_min
    else:
        lr = lr_min + 1 / 2 * (1 + math.cos((t - t_w) / (t_c - t_w) * math.pi)) * (
            lr_max - lr_min
        )

    return lr


def grad_clip(params: Iterable[nn.Parameter], max_grad_norm, eps: int = 1e-6):
    total_norm = 0
    for param in params:
        if param.grad is not None:
            norm = param.grad.norm()
            total_norm += norm**2  # 总梯度的L2范数是指每个部分的平方和再开方

    total_norm = torch.sqrt(total_norm)

    if total_norm > max_grad_norm:
        scale = max_grad_norm / (total_norm + eps)
    else:
        scale = 1

    for param in params:
        if param.grad is not None:
            param.grad.mul_(scale)
    return None


def get_batch_data(input_array, batch_size, context_length, device="mps"):

    max_vaild_index = len(input_array) - context_length - 1

    index = torch.randint(low=0, high=max_vaild_index + 1, size=(batch_size,))

    offset = torch.arange(context_length + 1)

    index = index.unsqueeze(-1) + offset

    input_array = torch.from_numpy(input_array)

    batch_data = input_array[index]

    x = batch_data[:, :-1]
    y = batch_data[:, 1:]

    if device is not None:
        x = x.to(device)
        y = y.to(device)

    return x, y


# checkpoint function
def save_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, obj
):
    checkpoint = {
        "model_weights": model.state_dict(),
        "optimizer_weights": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(checkpoint, obj)


def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model_weights = checkpoint["model_weights"]
    optimizer_weights = checkpoint["optimizer_weights"]

    model.load_state_dict(model_weights)
    optimizer.load_state_dict(optimizer_weights)

    return checkpoint["iteration"]
