import copy
from tqdm import tqdm
import torch
# from kdesbo.utils import generate_initial_data, sample_kde, qmc_sample_kde, update_edf, get_kernel_matrix
# from .utils import generate_initial_data, sample_kde, update_edf, get_kernel_matrix, sample_context_for_L_calculation
from botorch.models.gp_regression import SingleTaskGP
# import math
from botorch.optim.optimize import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import UpperConfidenceBound, ExpectedImprovement
from botorch.generation.gen import gen_candidates_scipy, TGenCandidates, gen_candidates_torch
# from botorch.models.transforms import Standardize, Normalize

from abc import abstractmethod

from botorch.utils.sampling import draw_sobol_samples
from mcmc_algorithms import eula_best, mala_best
def _bounds_to_tensor(bounds, dtype=torch.float64):
    '''
    Convert list of tuples bounds to (2, dim) tensor
    '''
    if isinstance(bounds, torch.Tensor):
        return bounds.to(dtype=dtype)
    lower = [b[0] for b in bounds]
    upper = [b[1] for b in bounds]
    return torch.tensor([lower, upper], dtype=dtype)


def generate_initial_data(problem, init_size, eps = 1e-6):
    '''
    :param problem: TestProblem
    :param init_size: number of initial evaluated data.
    :return: initial 'init_size' x 'dim' train_X, 'init_size' x 'dim' train_Y, train_Noise
    '''
    bounds = _bounds_to_tensor(problem._bounds, dtype=torch.float64)
    X = draw_sobol_samples(bounds=bounds, n=init_size, q=1).squeeze(-2)
    X = X.to(dtype=bounds.dtype, device=bounds.device)
    X = torch.clamp(X, min=bounds[0] + eps, max=bounds[1] - eps)
    Y = problem(X)
    return X, Y

class BOTorchOptimizer:
    def __init__(self, problem, init_size=10, running_rounds=200, device=torch.device('cpu'), beta=1.5):
        self.problem = problem
        self.init_size = init_size
        self.running_rounds = running_rounds
        self.train_X, self.train_Y = generate_initial_data(self.problem, self.init_size)
        self.train_X, self.train_Y = self.train_X.to(device), self.train_Y.to(device)

        self.beta = beta
        self.best_Y = torch.max(self.train_Y)
        self.cumulative_reward = None
        self.cumulative_regret = []
        self.device = device

        self.bounds = _bounds_to_tensor(problem._bounds).to(self.device)

    def get_model(self):
        X_contexts = self.train_X
        mean = self.train_Y.mean()
        sigma = self.train_Y.std()
        Y = (self.train_Y-mean)/sigma
        model = SingleTaskGP(X_contexts, Y.reshape(-1, 1)).to(device=self.device)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    def evaluate_new_candidates(self, candidates, i):
        candidates = candidates.cpu()
        new_Y= self.problem(candidates) # this calls the evaluate_true function via the forward function
        self.train_X = torch.cat((self.train_X, candidates), dim=0)
        self.train_Y = torch.cat((self.train_Y, new_Y), dim=0)
        if self.best_Y < new_Y.item():
            self.best_Y = float(new_Y.item())
        print(f"At running_rounds {i}, the best instantaneous value is :{self.best_Y}")
        if self.cumulative_reward is None:
            self.cumulative_reward = torch.sum(self.train_Y.reshape(-1))
        else:
            self.cumulative_reward += float(new_Y.item())
        self.cumulative_regret.append(self.best_Y)
        print(f"At running_rounds {i}, candidate :{candidates}, new_value :{new_Y}")

    @abstractmethod
    def run_opt(self):
        pass

class UCB_Optimizer(BOTorchOptimizer):
    def __init__(self, problem, init_size=10, running_rounds=150, device=torch.device('cpu'), beta=1.5):
        super().__init__(problem, init_size, running_rounds, device=device, beta=beta)

    def run_opt(self):
        for i in tqdm(range(self.init_size, self.running_rounds)):
            model = self.get_model()
# TODO: is acquisition function fixed?
            ucb = UpperConfidenceBound(
                model=model,
                beta=self.beta,
            )
            #print(self.train_X.shape)
            candidates, _ = optimize_acqf(
                acq_function=ucb,
                bounds=self.bounds[:, :(self.problem.dim-self.problem.contexts_dim)],
                q=1,
                num_restarts=10,
                raw_samples=1024,
                gen_candidates=gen_candidates_torch,
                options={"batch_limit": 512}
            )

            self.evaluate_new_candidates(candidates.detach(), i)
        return self.train_X, self.train_Y, self.cumulative_regret, self.best_Y


class HMC_sampling(BOTorchOptimizer):
    def __init__(self, problem, init_size=10, running_rounds=200, device=torch.device('cpu'), beta=1.5, 
                 burnin_step_size=None, step_size=None, n_burn_in_steps=None, n_iterations_steps=None, n_leapfrog_steps=None):
        
        super().__init__(problem, init_size, running_rounds, device=device, beta=beta)

        self.burnin_step_size = 0.05 if burnin_step_size is None else burnin_step_size
        self.step_size = 0.05 if step_size is None else step_size
        self.n_burn_in_steps = 100 if n_burn_in_steps is None else n_burn_in_steps
        self.n_iterations_steps = 200 if n_iterations_steps is None else n_iterations_steps
        self.n_leapfrog_steps = 10 if n_leapfrog_steps is None else n_leapfrog_steps

        self.acqf = None

    def _log_prob(self, x):
        return -self.acqf(x.unsqueeze(0)).sum()
    
    def _log_prob_gradient(self, x, eps=1e-5):
        gradient = torch.zeros_like(x)
        for i in range(len(x)):
            x_f = x.clone()
            x_b = x.clone()
            x_f[i] += eps
            x_b[i] -= eps
            gradient[i] = (self._log_prob(x_f) - self._log_prob(x_b)) / (2 * eps)
        return gradient
    
    def leapfrog(self, x, v, gradient, step_size, n_steps):
        v = v - 0.5 * step_size * gradient(x)
        for _ in range(n_steps):
            x = x + step_size * v
            v = v - step_size * gradient(x)
        x += step_size * v
        v -= 0.5 * step_size * gradient(x)
        return x, v

    def sample_step(self, x_old, step_size, n_steps):
        def E(x): return -self._log_prob(x)
        def gradient(x): return -self._log_prob_gradient(x)

        def K(v): return 0.5 * torch.sum(v**2)
        def H(x, v): return E(x) + K(v)

        v_old = torch.randn_like(x_old)

        x_new, v_new = self.leapfrog(x_old.clone(),
                                     v_old.clone(),
                                     gradient,
                                     step_size,
                                     n_steps
                                     )
        
        log_accept = min(0, -(H(x_new, v_new) - H(x_old, v_old)))
        if torch.log(torch.rand(1)) < log_accept:
            return True, x_new
        else:
            return False, x_old

    def sample(self):
        dim = self.bounds.shape[1]
        x0 = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * torch.rand(dim, device=self.bounds.device)

        x = x0.clone()
        chain = []
        for _ in range(self.n_burn_in_steps):
            _, x = self.sample_step(x, step_size=self.burnin_step_size, n_steps=self.n_leapfrog_steps)
        
        for _ in range(self.n_iterations_steps):
            _, x = self.sample_step(x, step_size=self.step_size, n_steps=self.n_leapfrog_steps)
            chain.append(x.clone())
        
        chain = torch.stack(chain)
        with torch.no_grad():
            acq_chain = self.acqf(chain.unsqueeze(1))
        
        best_x_idx = torch.argmax(acq_chain).item()
        best_x = chain[best_x_idx].unsqueeze(0) #as output should have shape (1, dim) to match previous optimize_acqf logic

        best_x = torch.clamp(best_x, self.bounds[0], self.bounds[1])

        return best_x
   

    def run_opt(self):
        for i in tqdm(range(self.init_size, self.running_rounds)):
            model = self.get_model()

            ucb = UpperConfidenceBound(
                model=model,
                beta=self.beta,
            )

            self.acqf = ucb #should acqf be passed into sampling from self. ?

            candidates = self.sample()
            self.evaluate_new_candidates(candidates.detach(), i)
        return self.train_X, self.train_Y, self.cumulative_regret, self.best_Y
class MALA_sampling(BOTorchOptimizer):
    def __init__(
        self,
        problem,
        init_size=10,
        running_rounds=200,
        device=torch.device("cpu"),
        beta=1.5,
        step_size=0.03,
        n_burn_in_steps=100,
        n_iterations_steps=200,
        temp=1.0,
        n_restarts=4,
        init_noise=0.05,
        target_acceptance=0.57,
        adapt_step=True,
    ):
        super().__init__(problem, init_size, running_rounds, device=device, beta=beta)

        self.step_size = float(step_size)
        self.n_burn_in_steps = int(n_burn_in_steps)
        self.n_iterations_steps = int(n_iterations_steps)
        self.temp = float(temp)
        self.min_step_size = 0.005
        self.max_step_size = 0.08
        self.n_restarts = int(n_restarts)
        self.init_noise = float(init_noise)
        self.target_acceptance = float(target_acceptance)
        self.adapt_step = bool(adapt_step)

        self.acqf = None

        self.last_acceptance_rate = None
        self.last_chain = None
        self.last_acq_vals = None

    def _log_pi(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x.view(1, 1, -1)
        return self.acqf(x_in).sum() / self.temp

    def _grad_log_pi(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().clone().requires_grad_(True)
        lp = self._log_pi(x)
        (g,) = torch.autograd.grad(lp, x, create_graph=False)
        return g.detach()

    @staticmethod
    def _log_q(x_to: torch.Tensor, mean: torch.Tensor, step_size: float) -> torch.Tensor:
        diff = x_to - mean
        return -0.5 * diff.pow(2).sum() / (step_size ** 2)

    def _mala_step(self, x_old: torch.Tensor, step_size: float):
        g_old = self._grad_log_pi(x_old)
        mean_fwd = x_old + 0.5 * (step_size ** 2) * g_old

        x_prop = mean_fwd + step_size * torch.randn_like(x_old)
        x_prop = torch.clamp(x_prop, self.bounds[0], self.bounds[1])

        g_prop = self._grad_log_pi(x_prop)
        mean_rev = x_prop + 0.5 * (step_size ** 2) * g_prop

        log_pi_old = self._log_pi(x_old)
        log_pi_prop = self._log_pi(x_prop)

        log_q_fwd = self._log_q(x_to=x_prop, mean=mean_fwd, step_size=step_size)
        log_q_rev = self._log_q(x_to=x_old, mean=mean_rev, step_size=step_size)

        log_alpha = (log_pi_prop + log_q_rev) - (log_pi_old + log_q_fwd)
        log_u = torch.log(torch.rand(1, device=x_old.device, dtype=x_old.dtype))

        if log_u < torch.minimum(log_alpha, torch.zeros_like(log_alpha)):
            return True, x_prop
        return False, x_old

    def _init_from_best(self) -> torch.Tensor:
        best_idx = torch.argmax(self.train_Y.view(-1)).item()
        x_best = self.train_X[best_idx].detach().clone().to(self.bounds.device)

        x0 = x_best + self.init_noise * torch.randn_like(x_best)
        x0 = torch.clamp(x0, self.bounds[0], self.bounds[1])
        return x0

    def _run_single_chain(self, x0: torch.Tensor):
        x = x0.clone()
        step_size = self.step_size

        accepted = 0
        total = 0

        # burn-in
        for _ in range(self.n_burn_in_steps):
            acc, x = self._mala_step(x, step_size)
            accepted += int(acc)
            total += 1

        burn_acc_rate = accepted / max(total, 1)

        # simple adaptive tuning after burn-in
        if self.adapt_step:
            if burn_acc_rate < self.target_acceptance - 0.15:
                step_size *= 0.7
            elif burn_acc_rate > self.target_acceptance + 0.15:
                step_size *= 1.2

        chain = []
        chain_acc = 0
        chain_total = 0

        for _ in range(self.n_iterations_steps):
            acc, x = self._mala_step(x, step_size)
            chain_acc += int(acc)
            chain_total += 1
            chain.append(x.clone())

        chain = torch.stack(chain)

        with torch.no_grad():
            acq_vals = self.acqf(chain.unsqueeze(1)).view(-1)

        best_idx = torch.argmax(acq_vals).item()
        best_x = chain[best_idx].view(1, -1)
        best_val = acq_vals[best_idx].item()

        acc_rate = chain_acc / max(chain_total, 1)
        return best_x, best_val, chain, acq_vals, acc_rate, step_size

    def sample(self) -> torch.Tensor:
        best_x_global = None
        best_val_global = -float("inf")
        best_chain = None
        best_acq_vals = None
        best_acc_rate = None

        for _ in range(self.n_restarts):
            x0 = self._init_from_best()
            best_x, best_val, chain, acq_vals, acc_rate, used_step = self._run_single_chain(x0)

            if best_val > best_val_global:
                best_val_global = best_val
                best_x_global = best_x
                best_chain = chain
                best_acq_vals = acq_vals
                best_acc_rate = acc_rate
                self.step_size = used_step  # softly carry tuned step forward

        self.last_chain = best_chain.detach().clone()
        self.last_acq_vals = best_acq_vals.detach().clone()
        self.last_acceptance_rate = best_acc_rate

        best_x_global = torch.clamp(best_x_global, self.bounds[0], self.bounds[1])
        return best_x_global

    def run_opt(self):
        for i in tqdm(range(self.init_size, self.running_rounds)):
            model = self.get_model()
            self.acqf = UpperConfidenceBound(model=model, beta=self.beta)

            candidates = self.sample()
            print(f"MALA acceptance rate: {self.last_acceptance_rate:.3f}, step_size: {self.step_size:.4f}")
            self.evaluate_new_candidates(candidates.detach(), i)

        return self.train_X, self.train_Y, self.cumulative_regret, self.best_Y
class ULA_sampling(BOTorchOptimizer):
    def __init__(
        self,
        problem,
        init_size=10,
        running_rounds=200,
        device=torch.device("cpu"),
        beta=1.5,
        step_size=0.005,
        n_burn_in_steps=100,
        n_iterations_steps=200,
        temp=1.0,
        n_restarts=4,
        init_noise=0.05,
    ):
        super().__init__(problem, init_size, running_rounds, device=device, beta=beta)

        self.step_size = float(step_size)
        self.n_burn_in_steps = int(n_burn_in_steps)
        self.n_iterations_steps = int(n_iterations_steps)
        self.temp = float(temp)

        self.n_restarts = int(n_restarts)
        self.init_noise = float(init_noise)

        self.acqf = None
        self.last_chain = None
        self.last_acq_vals = None

    def _log_pi(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x.view(1, 1, -1)
        return self.acqf(x_in).sum() / self.temp

    def _grad_log_pi(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().clone().requires_grad_(True)
        lp = self._log_pi(x)
        (g,) = torch.autograd.grad(lp, x, create_graph=False)
        return g.detach()

    def _ula_step(self, x: torch.Tensor) -> torch.Tensor:
        eps = self.step_size
        g = self._grad_log_pi(x)
        noise = torch.randn_like(x)

        x_new = x + eps * g + torch.sqrt(
            torch.tensor(2.0 * eps, device=x.device, dtype=x.dtype)
        ) * noise

        x_new = torch.clamp(x_new, self.bounds[0], self.bounds[1])
        return x_new

    def _init_from_best(self) -> torch.Tensor:
        best_idx = torch.argmax(self.train_Y.view(-1)).item()
        x_best = self.train_X[best_idx].detach().clone().to(self.bounds.device)

        x0 = x_best + self.init_noise * torch.randn_like(x_best)
        x0 = torch.clamp(x0, self.bounds[0], self.bounds[1])
        return x0

    def _run_single_chain(self, x0: torch.Tensor):
        x = x0.clone()

        for _ in range(self.n_burn_in_steps):
            x = self._ula_step(x)

        chain = []
        for _ in range(self.n_iterations_steps):
            x = self._ula_step(x)
            chain.append(x.clone())

        chain = torch.stack(chain)

        with torch.no_grad():
            acq_vals = self.acqf(chain.unsqueeze(1)).view(-1)

        best_idx = torch.argmax(acq_vals).item()
        best_x = chain[best_idx].view(1, -1)
        best_val = acq_vals[best_idx].item()

        return best_x, best_val, chain, acq_vals

    def sample(self) -> torch.Tensor:
        best_x_global = None
        best_val_global = -float("inf")
        best_chain = None
        best_acq_vals = None

        for _ in range(self.n_restarts):
            x0 = self._init_from_best()
            best_x, best_val, chain, acq_vals = self._run_single_chain(x0)

            if best_val > best_val_global:
                best_val_global = best_val
                best_x_global = best_x
                best_chain = chain
                best_acq_vals = acq_vals

        self.last_chain = best_chain.detach().clone()
        self.last_acq_vals = best_acq_vals.detach().clone()

        best_x_global = torch.clamp(best_x_global, self.bounds[0], self.bounds[1])
        return best_x_global

    def run_opt(self):
        for i in tqdm(range(self.init_size, self.running_rounds)):
            model = self.get_model()
            self.acqf = UpperConfidenceBound(model=model, beta=self.beta)

            candidates = self.sample()
            self.evaluate_new_candidates(candidates.detach(), i)

        return self.train_X, self.train_Y, self.cumulative_regret, self.best_Y

class ULA_sampling(BOTorchOptimizer):
    def __init__(
        self,
        problem,
        init_size=10,
        running_rounds=200,
        device=torch.device("cpu"),
        beta=1.5,
        step_size=0.005,
        n_burn_in_steps=100,
        n_iterations_steps=200,
        temp=1.0,
        n_restarts=6,
        init_noise=0.03,
        global_restart_prob=0.3,
    ):
        super().__init__(problem, init_size, running_rounds, device=device, beta=beta)

        self.step_size = float(step_size)
        self.n_burn_in_steps = int(n_burn_in_steps)
        self.n_iterations_steps = int(n_iterations_steps)
        self.temp = float(temp)

        self.n_restarts = int(n_restarts)
        self.init_noise = float(init_noise)
        self.global_restart_prob = float(global_restart_prob)

        self.acqf = None
        self.last_chain = None
        self.last_acq_vals = None

    def _log_pi(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x.view(1, 1, -1)
        return self.acqf(x_in).sum() / self.temp

    def _grad_log_pi(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().clone().requires_grad_(True)
        lp = self._log_pi(x)
        (g,) = torch.autograd.grad(lp, x, create_graph=False)
        return g.detach()

    def _ula_step(self, x: torch.Tensor, step_size: float) -> torch.Tensor:
        eps = step_size
        g = self._grad_log_pi(x)
        noise = torch.randn_like(x)

        x_new = x + eps * g + torch.sqrt(
            torch.tensor(2.0 * eps, device=x.device, dtype=x.dtype)
        ) * noise

        x_new = torch.clamp(x_new, self.bounds[0], self.bounds[1])
        return x_new

    def _init_from_best(self) -> torch.Tensor:
        best_idx = torch.argmax(self.train_Y.view(-1)).item()
        x_best = self.train_X[best_idx].detach().clone().to(self.bounds.device)

        x0 = x_best + self.init_noise * torch.randn_like(x_best)
        x0 = torch.clamp(x0, self.bounds[0], self.bounds[1])
        return x0

    def _init_random(self) -> torch.Tensor:
        dim = self.bounds.shape[1]
        x0 = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * torch.rand(
            dim, device=self.bounds.device, dtype=self.bounds.dtype
        )
        return x0

    def _run_single_chain(self, x0: torch.Tensor):
        x = x0.clone()
        step_size = self.step_size

        for _ in range(self.n_burn_in_steps):
            x = self._ula_step(x, step_size)

        chain = []
        for _ in range(self.n_iterations_steps):
            x = self._ula_step(x, step_size)
            chain.append(x.clone())

        chain = torch.stack(chain)

        with torch.no_grad():
            acq_vals = self.acqf(chain.unsqueeze(1)).view(-1)

        best_idx = torch.argmax(acq_vals).item()
        best_x = chain[best_idx].view(1, -1)
        best_val = acq_vals[best_idx].item()

        return best_x, best_val, chain, acq_vals

    def sample(self) -> torch.Tensor:
        best_x_global = None
        best_val_global = -float("inf")
        best_chain = None
        best_acq_vals = None

        for _ in range(self.n_restarts):
            if torch.rand(1).item() < self.global_restart_prob:
                x0 = self._init_random()
            else:
                x0 = self._init_from_best()

            best_x, best_val, chain, acq_vals = self._run_single_chain(x0)

            if best_val > best_val_global:
                best_val_global = best_val
                best_x_global = best_x
                best_chain = chain
                best_acq_vals = acq_vals

        self.last_chain = best_chain.detach().clone()
        self.last_acq_vals = best_acq_vals.detach().clone()

        best_x_global = torch.clamp(best_x_global, self.bounds[0], self.bounds[1])
        return best_x_global

    def run_opt(self):
        for i in tqdm(range(self.init_size, self.running_rounds)):
            model = self.get_model()
            self.acqf = UpperConfidenceBound(model=model, beta=self.beta)

            candidates = self.sample()
            self.evaluate_new_candidates(candidates.detach(), i)

        return self.train_X, self.train_Y, self.cumulative_regret, self.best_Y
