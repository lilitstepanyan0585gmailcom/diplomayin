import torch
import numpy as np
import math
from botorch.test_functions.synthetic import SyntheticTestFunction


class Ackley(SyntheticTestFunction):

    # The last dimension follows a Gaussian Distribution.
    _check_grad_at_opt = False

    def __init__(
        self,
        mu=0.5,
        sigma=0.2,
        dim=3,
        noise_std=None,
        negate=False,
        bounds=None,
    ):
        self.dim = dim
        self.contexts_dim = 0
        #self._bounds = [(-32.768, 32.768) for _ in range(self.dim)]
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        self.continuous_inds = list(range(self.dim))
        self.discrete_inds = []
        self.categorical_inds = []
        super().__init__(noise_std=noise_std, negate=False, bounds=bounds)
        self.a = 20
        self.b = 0.2
        self.c = 2 * math.pi
        self.negate_ = negate
        self.mu = mu
        self.sigma = sigma
        self.max_stochastic = 12.78
        if negate:
            self.max_stochastic = -12.78

    def _evaluate_true(self, X):
        X = X*65.536 - 32.768
        a, b, c = self.a, self.b, self.c
        part1 = -a * torch.exp(-b / math.sqrt(self.dim) * torch.norm(X, dim=-1))
        part2 = -(torch.exp(torch.mean(torch.cos(c * X), dim=-1)))
        f_X = part1 + part2 + a + math.e
        if self.negate_:
            f_X *= -1
        # return f_X
        return f_X.view(-1, 1) #for plotting and BoTorch?


class Hartmann(SyntheticTestFunction):
    def __init__(
        self,
        mu=0.5,
        sigma=0.2,
        dim=6,
        noise_std=None,
        negate=False,
        bounds=None,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        if dim not in (6,):
            raise ValueError(f"Hartmann with dim {dim} not defined")
        self.dim = dim
        self.contexts_dim = 0
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        self.continuous_inds = list(range(self.dim))
        self.discrete_inds = []
        self.categorical_inds = []
        super().__init__(noise_std=noise_std, negate=False, bounds=bounds)

        self.register_buffer("ALPHA", torch.tensor([1.0, 1.2, 3.0, 3.2]))
        if dim == 3:
            A = [[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]]
            P = [
                [3689, 1170, 2673],
                [4699, 4387, 7470],
                [1091, 8732, 5547],
                [381, 5743, 8828],
            ]
        elif dim == 4:
            A = [
                [10, 3, 17, 3.5],
                [0.05, 10, 17, 0.1],
                [3, 3.5, 1.7, 10],
                [17, 8, 0.05, 10],
            ]
            P = [
                [1312, 1696, 5569, 124],
                [2329, 4135, 8307, 3736],
                [2348, 1451, 3522, 2883],
                [4047, 8828, 8732, 5743],
            ]
        elif dim == 6:
            A = [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
            P = [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        self.register_buffer("A", torch.tensor(A, dtype=torch.float))
        self.register_buffer("P", torch.tensor(P, dtype=torch.float))
        self.negate_ = negate
        self.mu = mu
        self.sigma = sigma
        self.max_stochastic =  -2.3165 # [0.2185, 0.1531, 0.4913, 0.2764, 0.3199]
        if negate:
            self.max_stochastic = 2.3165 # [0.2185, 0.1531, 0.4913, 0.2764, 0.3199]

    def _evaluate_true(self, X):
        self.to(device=X.device, dtype=X.dtype)

        inner_sum = torch.sum(self.A * (X.unsqueeze(-2) - 0.0001 * self.P) ** 2, dim=-1)
        H = torch.sum(self.ALPHA * torch.exp(-inner_sum), dim=-1)
        f_X = -H
        if self.negate_:
            f_X *= -1
        return f_X.unsqueeze(-1)


class Modified_Branin(SyntheticTestFunction):

    # The last dimension follows a Gaussian Distribution.
    _check_grad_at_opt = False

    def __init__(
        self,
        mu=0.5,
        sigma=0.2,
        dim=4,
        noise_std=None,
        negate=False,
        bounds=None,
    ):
        self.dim = dim
        self.contexts_dim = 0
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        self.continuous_inds = list(range(self.dim))
        self.discrete_inds = []
        self.categorical_inds = []
        super().__init__(noise_std=noise_std, negate=False, bounds=bounds)
        self.negate_ = negate
        self.mu = mu
        self.sigma = sigma
        self.max_stochastic = 16.0649 # [0.1851, 0.2004]
        if negate:
            self.max_stochastic = -16.0649

    def _evaluate_true(self, X):
        u1, v1, u2, v2 = X[:, 0]*15-5, X[:, 1]*15, X[:, 2]*15-5, X[:, 3]*15,
        y1 = Branin(u1, v1)
        y2 = Branin(u2, v2)
        f_X = torch.sqrt(y1*y2)
        if self.negate_:
            f_X *= -1
        # return f_X
        return f_X.view(-1, 1) #for plotting and BoTorch?


def Branin(u, v):
    return torch.pow(v-5.1/(4*math.pi*math.pi) * u * u + 5.0/math.pi*u - 6.0, 2) +\
        10.0*(1-1/(8*math.pi))*torch.cos(u) + 10.0


class Hartmann_complicated(SyntheticTestFunction):
    def __init__(
        self,
        mu=None,
        sigma=None,
        dim=6,
        noise_std=None,
        negate=False,
        bounds=None,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        if dim not in (6,):
            raise ValueError(f"Hartmann with dim {dim} not defined")
        self.dim = dim
        self.contexts_dim = 0
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        self.continuous_inds = list(range(self.dim))
        self.discrete_inds = []
        self.categorical_inds = []
        super().__init__(noise_std=noise_std, negate=False, bounds=bounds)
        self.register_buffer("ALPHA", torch.tensor([1.0, 1.2, 3.0, 3.2]))
        if dim == 3:
            A = [[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]]
            P = [
                [3689, 1170, 2673],
                [4699, 4387, 7470],
                [1091, 8732, 5547],
                [381, 5743, 8828],
            ]
        elif dim == 4:
            A = [
                [10, 3, 17, 3.5],
                [0.05, 10, 17, 0.1],
                [3, 3.5, 1.7, 10],
                [17, 8, 0.05, 10],
            ]
            P = [
                [1312, 1696, 5569, 124],
                [2329, 4135, 8307, 3736],
                [2348, 1451, 3522, 2883],
                [4047, 8828, 8732, 5743],
            ]
        elif dim == 6:
            A = [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
            P = [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        self.register_buffer("A", torch.tensor(A, dtype=torch.float))
        self.register_buffer("P", torch.tensor(P, dtype=torch.float))
        self.negate_ = negate
        self.mu = mu
        self.sigma = sigma
        self.max_stochastic = -2.1
        if negate:
            self.max_stochastic = 2.1
        self.center = np.array([0.1, 0.3, 0.4, 0.5, 0.7, 0.8])
        self.std = np.array([0.02, 0.075, 0.1, 0.1, 0.075, 0.03])

    def _evaluate_true(self, X):
        self.to(device=X.device, dtype=X.dtype)
        inner_sum = torch.sum(self.A * (X.unsqueeze(-2) - 0.0001 * self.P) ** 2, dim=-1)
        H = (torch.sum(self.ALPHA * torch.exp(-inner_sum), dim=-1))
        f_X = -H
        if self.negate_:
            f_X *= -1
        # return f_X
        return f_X.view(-1, 1) #for plotting and BoTorch?


class Continuous_Vendor(SyntheticTestFunction):
    _check_grad_at_opt = False

    def __init__(
        self,
        mu=2,
        sigma=20,
        dim=2,
        noise_std=None,
        negate=False,
        bounds=None,
    ):
        self.dim = dim
        self.contexts_dim = 0
        #self._bounds = [(0.0, 10.0) for _ in range(self.dim)]
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        self.continuous_inds = list(range(self.dim))
        self.discrete_inds = []
        self.categorical_inds = []
        super().__init__(noise_std=noise_std, negate=False, bounds=bounds)
        self.negate_ = negate
        self.mu = mu
        self.max_stochastic = -0.464
        self.cost = 5.0
        self.price = 9.0
        self.salvage_price = 1.0
        if negate:
            self.max_stochastic = 0.464

    def _evaluate_true(self, X):
        deterministic_demand = torch.full_like(X, self.mu)
        # f_X = self.price*torch.min(context_clamp,X) - self.cost*X + self.salvage_price*torch.max(torch.tensor(0),X-context_clamp)
        f_X = (self.price*torch.minimum(deterministic_demand, X) - 
               self.cost*X + 
               self.salvage_price*torch.max(torch.tensor(0),X-deterministic_demand))
        f_X = f_X.mean(dim=-1) #TODO: спросить устраивает ли среднее по дименшенам или мы хотим что-то другое
        f_X = f_X*(-1)
        if self.negate_:
            f_X *= -1
        # return f_X.reshape(-1,)
        return f_X.view(-1, 1) #for plotting and BoTorch?


class SixHumpCamel(SyntheticTestFunction):
    _check_grad_at_opt = False

    def __init__(
        self,
        dim=2,
        noise_std=None,
        negate=False,
        bounds=None,
    ):
        self.dim = dim
        self.contexts_dim = 0
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)] # normalized bounds
        self.continuous_inds = list(range(self.dim))
        self.discrete_inds = []
        self.categorical_inds = []
        super().__init__(noise_std=noise_std, negate=False, bounds=bounds)
        self.max_stochastic = 3.0164  # Global minimum value
        if negate:
            self.max_stochastic = -3.0164

    def _evaluate_true(self, X):
        # scale to original bounds X_1 [-2, 2], X_2 [-1, 1] Reduced domain
        X = X * torch.tensor([4.0, 2.0]) - torch.tensor([2.0, 1.0])
        x1, x2 = X[..., 0], X[..., 1]
        f_X = -1*((4 - 2.1 * x1**2 + x1**4 / 3) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2)
        if self.negate_:
            f_X *= -1
        # return f_X
        return f_X.view(-1, 1) #for plotting and BoTorch?


class ThreeHumpCamel(SyntheticTestFunction):
    _check_grad_at_opt = False

    def __init__(
        self,
        mu=0.5,
        sigma=0.2,
        noise_std=None,
        negate=False,
        bounds=None,
    ):
        self.dim = 2
        self.contexts_dim = 0
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        self.continuous_inds = list(range(self.dim))
        self.discrete_inds = []
        self.categorical_inds = []
        super().__init__(noise_std=noise_std, negate=False, bounds=bounds)
        self.max_stochastic = 0.1564
        if negate:
            self.max_stochastic = -0.1564
        self.mu = mu
        self.sigma = sigma

        self.negate_ = negate

    def _evaluate_true(self, X):
        X = X * torch.tensor([2.0, 2.0]) - torch.tensor([1.0, 1.0])
        x1, x2 = X[..., 0], X[..., 1]
        f_X = 2 * x1**2 - 1.05 * x1**4 + x1**6 / 6 + x1 * x2 + x2**2
        f_X = f_X*(-1)
        if self.negate_:
            f_X *= -1
        # return f_X
        return f_X.view(-1, 1) #for plotting and BoTorch?


class StyblinskiTang(SyntheticTestFunction):
    _check_grad_at_opt = False

    def __init__(
            self,
            dim=2,
            bounds=None,
            noise_std=None,
            negate=False
    ):
        self.dim = dim
        self.contexts_dim = 0
        # self._bounds = [(-5.0, 5.0) for _ in range(self.dim)] - original bounds
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        self.continuous_inds = list(range(self.dim))
        self.discrete_inds = []
        self.categorical_inds = []
        super().__init__(noise_std=noise_std, negate=False, bounds=bounds)
        self.negate_ = negate

    def _evaluate_true(self, X):
        X = -5 + X * 10
        f_X = 0.5 * (X**4 - 16 * X**2 + 5 * X).sum(dim=-1)
        if self.negate_:
            f_X *= -1
        return f_X.view(-1, 1)

class Rosenbrock(SyntheticTestFunction):
    _check_grad_at_opt = False

    def __init__(self, dim=2, noise_std=None, negate=False, bounds=None):
        self.dim = dim
        self.contexts_dim = 0
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]

        self.continuous_inds = list(range(self.dim))
        self.discrete_inds = []
        self.categorical_inds = []

        super().__init__(noise_std=noise_std, negate=False, bounds=bounds)

        self.negate_ = negate
        self.can_calculate_stochastic = False

    def _evaluate_true(self, X):
        Xs = X * 4.0 - 2.0
        x0 = Xs[..., :-1]
        x1 = Xs[..., 1:]
        f_X = torch.sum(100.0 * (x1 - x0**2) ** 2 + (1.0 - x0) ** 2, dim=-1)
        if self.negate_:
            f_X *= -1
        return f_X
