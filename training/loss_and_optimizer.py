import torch
from torch import optim
from torch import Tensor
from jaxtyping import Float, Int
from collections.abc import Callable, Iterable
from typing import Optional
from math import cos, pi


def cross_entropy(
        inputs: Float[Tensor, "batch_size sequence_length vocab_size"] | Float[Tensor, "batch_size vocab_size"], 
        targets: Int[Tensor, "batch_size sequence_length"] | Int[Tensor, "batch_size"]
) -> Float[Tensor, ""]:
    """
    inputs: Logits tensor. Can be either:
        - (batch_size, vocab_size) for single prediction per example
        - (batch_size, sequence_length, vocab_size) for sequence prediction
    targets: Target indices. Shape should match the non-vocab dimensions of inputs.
    """
    # Handle both 2D and 3D inputs by reshaping to 2D for computation
    if inputs.dim() == 3:
        # Reshape (batch_size, sequence_length, vocab_size) -> (batch_size * sequence_length, vocab_size)
        batch_size, sequence_length, vocab_size = inputs.shape
        inputs_2d = inputs.view(-1, vocab_size)
        targets_1d = targets.view(-1)
    else:
        # Already 2D
        inputs_2d = inputs
        targets_1d = targets
    
    max_logits, _ = inputs_2d.max(dim=1, keepdim=True) # [N, 1]
    # calculate log(sum(exp(x - max))) + max, for numerical stability
    log_sum_exp = torch.log(torch.sum(torch.exp(inputs_2d - max_logits), dim=1)) + max_logits.squeeze(1) # [N]
    correct_logit = inputs_2d[torch.arange(inputs_2d.size(0)), targets_1d]
    loss = - (correct_logit - log_sum_exp)
    return loss.mean()


def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int
) -> float:
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.
    """
    if it < warmup_iters:
        return it * max_learning_rate / warmup_iters
    elif warmup_iters <= it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + cos((it-warmup_iters)*pi/(cosine_cycle_iters-warmup_iters))) * (max_learning_rate-min_learning_rate)
    else:
        return min_learning_rate


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float=1e-6) -> None:
    """
    Clips gradients by global norm across all parameters.
    """
    # Convert to list to ensure we can iterate multiple times
    params_with_grad = [p for p in parameters if p.grad is not None]
    if len(params_with_grad) == 0:
        return
    with torch.no_grad():
        # Calculate the global L2 norm across all parameter gradients
        # Initialize on the same device as the first parameter
        device = params_with_grad[0].device if params_with_grad else torch.device('cpu')
        total_norm_squared = torch.tensor(0.0, device=device)
        for p in params_with_grad:
            if p.grad is not None:
                total_norm_squared += p.grad.norm() ** 2
        total_norm = torch.sqrt(total_norm_squared)
        # Apply clipping if the global norm exceeds max_l2_norm
        if total_norm > max_l2_norm:
            clip_coef = max_l2_norm / (total_norm + eps)
            for p in params_with_grad:
                if p.grad is not None:
                    p.grad.mul_(clip_coef)


class AdamW(optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter],
                betas: tuple[Float, Float]=(0.9, 0.999),
                lr=1e-3, 
                eps=1e-8,
                weight_decay=0.01):
        defaults: dict = {"betas": betas, "lr": lr, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                grad = p.grad.data
                # Initialization
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                m, v = state["m"], state["v"]
                state["t"] += 1
                t = state["t"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)          #m=β1+(1-β1)g
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)#v=β2+(1-β2)g^2

                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
  
                p.data.mul_(1 - lr * weight_decay) # Decoupled weight decay
                p.data.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

        return loss
