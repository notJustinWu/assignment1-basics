from collections.abc import Callable, Iterable
from typing import Optional
import torch
import numpy as np
import math

class Adam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas = (0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] 
            beta = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data  
                state = self.state[p] 
                m = state.get("m", 0)
                v = state.get("v", 0)
                t = state.get("t", 0) + 1
                m = beta[0] * m + (1 - beta[0]) * g
                v = beta[1] * v + (1 - beta[1]) * (g ** 2)
                alpha_t = lr * ((1 - (beta[1] ** t)) ** 0.5)/(1 - (beta[0] ** t))
                p.data -= alpha_t * (m/(v ** 0.5 + eps))
                p.data -= lr * weight_decay * p.data

                state["t"] = t
                state["m"] = m 
                state["v"] = v
        return loss

def learning_rate_schedule(it: int, max_lr, min_lr, T_w, T_c):
    if it < T_w:
        a_t = (it/T_w) * max_lr
    elif T_w <= it <= T_c:
        inside = ((it - T_w)/(T_c - T_w)) * np.pi
        a_t = min_lr + 0.5 * (1 + np.cos(inside)) * (max_lr - min_lr)
    else:
        a_t = min_lr
    return a_t

def gradient_clipping(params: Iterable[torch.nn.Parameter], max_l2_norm: float):
    eps = 1e-6
    grads = []
    for p in params:
        if p.grad is not None:
            grads.append(p.grad.data.view(-1))
    if not grads:
        return 
    flat_grads = torch.cat(grads)
    norm = torch.norm(flat_grads, p = 2)
    if norm >= max_l2_norm:
        for p in params:
            if p.grad is not None:
                p.grad.data = p.grad.data * max_l2_norm/(norm + eps)
                
    



