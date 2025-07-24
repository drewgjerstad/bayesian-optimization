"""
This module is used to test Bayesian optimization primitives on the Branin
problem (typically embedded in higher-dimensions). Note that this is based on
the BoTorch tutorial for SAASBO (extended for other purposes).
"""

# Load Dependencies
import os
import torch
from torch.quasirandom import SobolEngine
from botorch import fit_fully_bayesian_model_nuts
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.test_functions import Branin
from plotting import parity_plot, progress_plot

from botorch.acquisition.logei import qLogExpectedImprovement

SMOKE_TEST = os.environ.get("SMOKE_TEST")

# Tensor Keyword Arguments
tkwargs = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.double
}

def branin_embed(x):
    """
    `x` is assumed to be in [0, 1]^d
    """
    branin = Branin().to(**tkwargs)
    lower, upper = branin.bounds
    return branin(lower + (upper - lower) * x[..., :2])

#===============================================================================
def main():
    """
    This function can be used to test various primitives on the Branin function.
    """

    # Set MC Parameters (warmup, number of samples, thinning)
    WARMUP_STEPS = 256 if not SMOKE_TEST else 32
    NUM_SAMPLES = 128 if not SMOKE_TEST else 16
    THINNING = 16

    # Define Problem Dimensions
    DIM = 30 if not SMOKE_TEST else 2

    # Define Evaluation Budget
    N_INITIAL = 10
    N_ITERATIONS = 8 if not SMOKE_TEST else 1
    BATCH_SIZE = 5 if not SMOKE_TEST else 1
    print(f"Using a total of {N_INITIAL + BATCH_SIZE * N_ITERATIONS}"
          " function evaluations.")

    # Initialize Datasets for Training and Evaluating Using the Branin Function
    train_X = SobolEngine(dimension=DIM, scramble=True, seed=0).draw(N_INITIAL).to(**tkwargs)
    test_X = SobolEngine(dimension=DIM, scramble=True, seed=1).draw(N_INITIAL).to(**tkwargs)
    train_Y = branin_embed(train_X).unsqueeze(-1)
    test_Y = branin_embed(test_X).unsqueeze(-1)

    # Best Initial Points
    best_train = train_Y.min().item()
    best_test = test_Y.min().item()
    print(f"Best Initial (train) Point: {best_train:.3f}")
    print(f"Best Initial (test) Point: {best_test:.3f}")

    # Optimization Loop
    for i in range(N_ITERATIONS):
        # Flip Sign to Minimize the Function
        train_Y_neg = -1 * train_Y

        # Initialize GP Model
        gp_model = SaasFullyBayesianSingleTaskGP(
            train_X=train_X,
            train_Y=train_Y_neg,
            train_Yvar=torch.full_like(train_Y_neg, 1e-6),
            outcome_transform=Standardize(m=1),
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
        acquisition = qLogExpectedImprovement(
            model=gp_model,
            best_f=train_Y_neg.max()
        )
        candidates, _ = optimize_acqf(
            acquisition,
            bounds=torch.cat((torch.zeros(1, DIM), torch.ones(1, DIM))),
            q=BATCH_SIZE,
            num_restarts=10,
            raw_samples=1024,
        )

        # Get Next Set of Observations
        Y_next = torch.cat(
            [branin_embed(x).unsqueeze(-1) for x in candidates]
        ).unsqueeze(-1)

        if Y_next.min() < train_Y.min():
            idx_best = Y_next.argmin()
            x0, x1 = candidates[idx_best, :2].tolist()
            print(
                f"{i+1}) New Best: {Y_next[idx_best].item():.3f} @ "
                f"[{x0:.3f}, {x1:.3f}]"
            )

        # Augment Dataset with New Observations
        train_X = torch.cat((train_X, candidates))
        train_Y = torch.cat((train_Y, Y_next))

    # Plot Parity
    with torch.no_grad():
        posterior = gp_model.posterior(test_X)
    parity_plot(test_Y * -1, posterior)

    # Plot Progress
    progress_plot(train_Y, optimum=train_Y.min().item(), maximize=False,
                  title=f"Branin, D = {DIM}")


if __name__ == "__main__":
    main()
