import torch
from torch import nn


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        """
        父类的init方法会将传入的参数保存为self.param_groups
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            beta1 = betas[0]
            beta2 = betas[1]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:  # 对于每一个参数
                if p.grad is None:
                    continue

                state = self.state[p]
                grad = p.grad.data  # data是获取实际的数据
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                t = state["t"] + 1
                state["t"] = t
                m = state["m"]
                v = state["v"]
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).add_(grad**2, alpha=1 - beta2)

                # 修正后的一阶矩和二阶矩
                m_corr = m / (1 - beta1**t)
                v_corr = v / (1 - beta2**t)

                p.data.addcdiv_(m_corr, (v_corr.sqrt() + eps), value=-lr)

                p.data.mul_(1 - lr * weight_decay)

        return loss
