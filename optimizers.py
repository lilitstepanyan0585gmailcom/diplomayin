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
class Langevin_sampling(BOTorchOptimizer):
    def __init__(
        self,
        problem,
        init_size=10,
        running_rounds=200,
        device=torch.device("cpu"),
        beta=1.5,
        method="mala",          # "eula" or "mala"
        temperature=0.5,
        step=0.2,
        n_steps=3000,
        burn=300,
        thin=5,
        n_chains=16,
        use_reflect=True,
        seed=42,
    ):
        super().__init__(problem, init_size, running_rounds, device=device, beta=beta)
        self.method = method
        self.temperature = temperature
        self.step = step
        self.n_steps = n_steps
        self.burn = burn
        self.thin = thin
        self.n_chains = n_chains
        self.use_reflect = use_reflect
        self.seed = seed
        self.acqf = None

    def sample(self):
        if self.acqf is None:
            raise RuntimeError("acqf is not set")

        if self.method == "eula":
            res = eula_best(
                acqf=self.acqf,
                bounds=self.bounds,
                n_steps=self.n_steps,
                burn=self.burn,
                thin=self.thin,
                step=self.step,
                temperature=self.temperature,
                n_chains=self.n_chains,
                use_reflect=self.use_reflect,
                seed=self.seed,
            )
            return res.best_x

        if self.method == "mala":
            res = mala_best(
                acqf=self.acqf,
                bounds=self.bounds,
                n_steps=self.n_steps,
                burn=self.burn,
                thin=self.thin,
                step=self.step,
                temperature=self.temperature,
                n_chains=self.n_chains,
                use_reflect=self.use_reflect,
                seed=self.seed,
                adapt_step=True,
            )
            # print("MALA accept:", res.extra["accept_rate"], "step:", res.extra["final_step"])
            return res.best_x

        raise ValueError(f"Unknown method: {self.method}")

    def run_opt(self):
        for i in tqdm(range(self.init_size, self.running_rounds)):
            model = self.get_model()
            ucb = UpperConfidenceBound(model=model, beta=self.beta)
            self.acqf = ucb
            candidates = self.sample()
            self.evaluate_new_candidates(candidates.detach(), i)

        return self.train_X, self.train_Y, self.cumulative_regret, self.best_Y
