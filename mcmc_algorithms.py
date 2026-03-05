import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict

import torch

Tensor = torch.Tensor


# Bounds utilities
def _as_2d_bounds(bounds: Tensor, device=None, dtype=None) -> Tuple[Tensor, Tensor]:
    """
    bounds: torch.Tensor of shape (2, d)
    returns: (lb, ub) each of shape (1, d)
    """
    if not isinstance(bounds, torch.Tensor):
        raise TypeError("bounds must be a torch.Tensor of shape (2, d)")
    b = bounds.to(device=device or bounds.device, dtype=dtype or bounds.dtype)
    if b.ndim != 2 or b.shape[0] != 2:
        raise ValueError(f"bounds must have shape (2, d), got {tuple(b.shape)}")
    lb = b[0].view(1, -1)
    ub = b[1].view(1, -1)
    return lb, ub


def clamp_to_bounds(x: Tensor, lb: Tensor, ub: Tensor) -> Tensor:
    return torch.max(torch.min(x, ub), lb)


def reflect_to_bounds(x: Tensor, lb: Tensor, ub: Tensor) -> Tensor:
    """
    Reflection on box boundaries. Vectorized for (..., d).
    """
    w = ub - lb
    y = (x - lb) % (2.0 * w)
    y = torch.where(y <= w, y, 2.0 * w - y)
    return lb + y

# Acquisition wrappers
def make_acq_eval(acqf: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    """
    BoTorch acquisition functions typically expect X with shape (batch, q, d).
    Here we work with x of shape (n, d), and evaluate acq at X=(n, 1, d).
    Returns: (n,) tensor
    """
    def eval_acq(x_nd: Tensor) -> Tensor:
        v = acqf(x_nd.unsqueeze(1))  # (n,1,...) -> squeeze -> (n,)
        return v.squeeze(-1).squeeze(-1)
    return eval_acq


def grad_acq_autograd(acq_eval: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    """
    Computes gradient of acq w.r.t x via torch autograd.
    x: (n, d) -> grad: (n, d)
    NOTE: We explicitly enable grads here to avoid failures if outer code uses no_grad.
    """
    with torch.enable_grad():
        x_req = x.detach().requires_grad_(True)
        vals = acq_eval(x_req)  # (n,)
        (g,) = torch.autograd.grad(vals.sum(), x_req, retain_graph=False, create_graph=False)
        return g.detach()

# Result container
@dataclass
class MCMCBest:
    best_x: Tensor          #
    best_acq: float
    extra: Dict[str, float]


# EULA / ULA
def eula_best(
    acqf: Callable[[Tensor], Tensor],
    bounds: Tensor,
    n_steps: int = 3000,
    burn: int = 300,
    thin: int = 5,
    step: float = 0.20,          # epsilon
    temperature: float = 0.5,    # T
    n_chains: int = 16,
    use_reflect: bool = True,
    seed: Optional[int] = None,
) -> MCMCBest:
    """
    EULA/ULA targeting pi(x) ∝ exp(acq(x)/T). No MH correction.
    Returns the single best visited point as (1, d).
    """
    if seed is not None:
        torch.manual_seed(seed)

    device = bounds.device
    dtype = bounds.dtype
    lb, ub = _as_2d_bounds(bounds, device=device, dtype=dtype)

    acq_eval = make_acq_eval(acqf)
    d = lb.shape[-1]
    x = lb + (ub - lb) * torch.rand(n_chains, d, device=device, dtype=dtype)
    drift_scale = (step * step) / (2.0 * float(temperature))

    best_acq = -float("inf")
    best_x = x[0:1].clone()

    for k in range(n_steps):
        g = grad_acq_autograd(acq_eval, x)  # (n, d)

        # Update without tracking grads
        with torch.no_grad():
            noise = torch.randn_like(x) * step
            x = x + drift_scale * g + noise
            x = reflect_to_bounds(x, lb, ub) if use_reflect else clamp_to_bounds(x, lb, ub)

        if k >= burn and ((k - burn) % thin == 0):
            with torch.no_grad():
                a = acq_eval(x)  # (n,)
                idx = torch.argmax(a).item()
                val = float(a[idx].item())
                if val > best_acq:
                    best_acq = val
                    best_x = x[idx:idx+1].clone()

    best_x = clamp_to_bounds(best_x, lb, ub)
    return MCMCBest(best_x=best_x, best_acq=best_acq, extra={})


# -----------------------------
# MALA
# -----------------------------
def _log_q_gaussian(y: Tensor, mean: Tensor, step: float) -> Tensor:
    """
    q(y|x) = N(mean, step^2 I). returns (n,)
    """
    s2 = step * step
    d = y.shape[-1]
    diff2 = ((y - mean) ** 2).sum(dim=-1)
    return -0.5 * (diff2 / s2) - 0.5 * d * math.log(2.0 * math.pi * s2)


def mala_best(
    acqf: Callable[[Tensor], Tensor],
    bounds: Tensor,
    n_steps: int = 3000,
    burn: int = 300,
    thin: int = 5,
    step: float = 0.20,
    temperature: float = 0.5,
    n_chains: int = 16,
    use_reflect: bool = True,
    seed: Optional[int] = None,
    adapt_step: bool = True,
    adapt_start: int = 100,
    adapt_every: int = 50,
    target_accept: float = 0.57,
    adapt_strength: float = 0.05,
) -> MCMCBest:
    """
    MALA targeting pi(x) ∝ exp(acq(x)/T) with MH correction.
    Returns the single best visited point as (1, d).

    extra includes:
      - accept_rate
      - final_step
    """
    if seed is not None:
        torch.manual_seed(seed)

    device = bounds.device
    dtype = bounds.dtype
    lb, ub = _as_2d_bounds(bounds, device=device, dtype=dtype)

    acq_eval = make_acq_eval(acqf)
    d = lb.shape[-1]

    x = lb + (ub - lb) * torch.rand(n_chains, d, device=device, dtype=dtype)

    step_now = float(step)
    acc = 0
    prop = 0

    best_acq = -float("inf")
    best_x = x[0:1].clone()

    for k in range(n_steps):
        # grad log pi(x) = (1/T) grad a(x)
        g_x = grad_acq_autograd(acq_eval, x) / float(temperature)
        mean_fwd = x + 0.5 * (step_now * step_now) * g_x

        with torch.no_grad():
            y = mean_fwd + step_now * torch.randn_like(x)
            y = reflect_to_bounds(y, lb, ub) if use_reflect else clamp_to_bounds(y, lb, ub)

        lp_x = acq_eval(x) / float(temperature)
        lp_y = acq_eval(y) / float(temperature)

        # reverse proposal needs grad at y
        g_y = grad_acq_autograd(acq_eval, y) / float(temperature)
        mean_rev = y + 0.5 * (step_now * step_now) * g_y

        log_q_y_given_x = _log_q_gaussian(y, mean_fwd, step_now)
        log_q_x_given_y = _log_q_gaussian(x, mean_rev, step_now)

        log_alpha = (lp_y + log_q_x_given_y) - (lp_x + log_q_y_given_x)
        alpha = torch.exp(torch.clamp(log_alpha, max=0.0))

        with torch.no_grad():
            u = torch.rand_like(alpha)
            accept = u < alpha
            x = torch.where(accept.unsqueeze(-1), y, x)

        prop += n_chains
        acc += int(accept.sum().item())

        if adapt_step and (k >= adapt_start) and ((k + 1) % adapt_every == 0):
            acc_rate = acc / max(1, prop)
            step_now *= math.exp(adapt_strength * (acc_rate - target_accept))
            step_now = float(max(1e-4, min(step_now, 2.0)))

        if k >= burn and ((k - burn) % thin == 0):
            a = acq_eval(x)
            idx = torch.argmax(a).item()
            val = float(a[idx].item())
            if val > best_acq:
                best_acq = val
                best_x = x[idx:idx+1].clone()

    best_x = clamp_to_bounds(best_x, lb, ub)
    accept_rate = acc / max(1, prop)

    return MCMCBest(
        best_x=best_x,
        best_acq=best_acq,
        extra={"accept_rate": float(accept_rate), "final_step": float(step_now)},
    )
