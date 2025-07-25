"""
This module is used to test Bayesian optimization primitives on the Branin
problem (typically embedded in higher-dimensions). Note that this is based on
the BoTorch tutorial for SAASBO (extended for other purposes).
"""

# Load Dependencies
import torch
from torch.quasirandom import SobolEngine
from botorch import fit_fully_bayesian_model_nuts
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.test_functions import Branin

from botorch.acquisition.logei import qLogExpectedImprovement

from plotting import parity_plot, progress_plot

def branin_function(x:torch.Tensor, **tkwargs) -> torch.Tensor:
    """
    This function computes the output of the Branin function at `x`. Note that
    `x` is assumed to be in [0, 1]^d. Then, we will use `lower` and `upper` to
    unnormalize the inputs to the true domain of Branin before evaluating it.

    Args:
        x (torch.Tensor):
            Locations to evaluate the Branin function at.
        **tkwargs (dict):
            Tensor keyword arguments.
    
    Returns:
        Branin function evaluations at `x`.
    """

    branin = Branin().to(**tkwargs)
    lower, upper = branin.bounds
    return branin(lower + (upper - lower) * x[..., :2])


def get_sobol_datasets(dimension:int, n_initial:int, n_test:int,
                       seed:int=0, **tkwargs:dict) -> tuple:
    """
    This function will obtain training and evaluation datasets using Sobol
    sampling to select Branin function locations.

    Args:
        dimension (int):
            Dimension of the datasets.
        n_initial (int):
            Number of initial data points.
        n_test (int):
            Number of testing data points.
        seed (int, default=0):
            Seed for Sobol sampling (to enable reproducibility).
        **tkwargs (dict):
            Tensor keyword arguments.

    Returns:
        train_X (torch.Tensor):
            Training locations.
        test_X (torch.Tensor):
            Testing locations.
        train_Y (torch.Tensor):
            Training values.
        test_Y (torch.Tensor):
            Testing values.
    """

    # Sobol Engines
    sobol_train = SobolEngine(dimension=dimension, scramble=True, seed=seed)
    sobol_test = SobolEngine(dimension=dimension, scramble=True, seed=seed + 1)

    # Create Datasets
    train_X = sobol_train.draw(n_initial).to(**tkwargs)
    test_X = sobol_test.draw(n_test).to(**tkwargs)
    train_Y = branin_function(train_X).unsqueeze(-1)
    test_Y = branin_function(test_X).unsqueeze(-1)

    return train_X, test_X, train_Y, test_Y


def bayes_opt_loop(datasets:tuple, mc_params:tuple, eval_budget_params:tuple,
                   dim:int, acquisition_function) -> tuple:
    """
    This function runs a Bayesian optimization loop. It will return a tuple
    containing the final datasets acquired during optimization.

    Args:
        datasets (tuple):
            Tuple of datasets.
        mc_params (tuple):
            Tuple of Monte Carlo hyperparameters.
        eval_budget_params (tuple):
            Tuple of evaluation budget hyperparameters.
        dim (int):
            Problem dimension.
        acquisition_function:
            Acquisition function.

    Returns:
        final_datasets (tuple):
            Tuple of final datasets returned after optimization.
    """

    # Extract Datasets
    train_X, test_X, train_Y, test_Y = datasets

    # Extract Monte Carlo Hyperparameters
    WARMUP_STEPS, NUM_SAMPLES, THINNING = mc_params

    # Extract Evaluation Budget Hyperparameters
    N_INITIAL, N_ITERATIONS, BATCH_SIZE = eval_budget_params

    # Optimization Loop
    for i in range(N_ITERATIONS):
        # Flip Sign to Minimize the Function
        train_Y_neg = -1 * train_Y

        # Initialize GP Model
        gp_model = SaasFullyBayesianSingleTaskGP(
            train_X=train_X,
            train_Y=train_Y_neg,
            train_Yvar=torch.full_like(train_Y_neg, 1e-6),
            outcome_transform=Standardize(m=1)
        )

        # Fit the GP Model
        fit_fully_bayesian_model_nuts(
            gp_model,
            warmup_steps=WARMUP_STEPS,
            num_samples=NUM_SAMPLES,
            thinning=THINNING,
            disable_progbar=True,
        )

        # Optimize Acquisition Function
        candidates, _ = optimize_acqf(
            acquisition_function(model=gp_model, best_f=train_Y_neg.max()),
            bounds=torch.cat((torch.zeros(1, dim), torch.ones(1, dim))),
            q=BATCH_SIZE,
            num_restarts=10,
            raw_samples=1024,
        )

        # Get Next Set of Observations
        Y_next = torch.cat(
            [branin_function(x).unsqueeze(-1) for x in candidates]
        ).unsqueeze(-1)

        if Y_next.min() < train_Y.min():
            idx_best = Y_next.argmin()
            x0, x1 = candidates[idx_best, :2].tolist()
            print(f"{i+1}) New Best: {Y_next[idx_best].item():.3f} @ "
                  f"[{x0:.3f}, {x1:.3f}]")
            
        # Augment Dataset with New Observations
        train_X = torch.cat((train_X, candidates))
        train_Y = torch.cat((train_Y, Y_next))

    return train_X, test_X, train_Y, test_Y


#===============================================================================
def main():
    """
    Test different acquisition functions on the Branin problem.
    """

    # Tensor Keyword Arguments
    tkwargs = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "dtype": torch.double
    }

    # Monte Carlo Hyperparameters
    WARMUP_STEPS = 256
    NUM_SAMPLES = 128
    THINNING = 16

    # Problem Dimensions
    DIM = 30

    # Evaluation Budget Hyperparameters
    N_INITIAL = 10
    N_ITERATIONS = 8
    BATCH_SIZE = 5

    # Generate Datasets
    train_X, test_X, train_Y, test_Y = get_sobol_datasets(
        DIM, N_INITIAL, 50, 1234, **tkwargs)

    # Identify Best Initial Points
    best_train = train_Y.min().item()
    best_test = test_Y.min().item()
    print(f"Best Initial (train) Point: {best_train:.3f}")
    print(f"Best Initial (test) Point: {best_test:.3f}")

    # Optimization Loop
    train_X, test_X, train_Y, test_Y = bayes_opt_loop(
        datasets=(train_X, test_X, train_Y, test_Y),
        mc_params=(WARMUP_STEPS, NUM_SAMPLES, THINNING),
        eval_budget_params=(N_INITIAL, N_ITERATIONS, BATCH_SIZE),
        dim=DIM,
        acquisition_function=qLogExpectedImprovement
    )

    # Plot Progress
    progress_plot(train_Y, optimum=train_Y.min().item(), maximize=False,
                  title=f"Branin, D = {DIM}")


if __name__ == "__main__":
    main()
