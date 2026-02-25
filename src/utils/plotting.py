"""
This module contains methods for plotting various results and functions common
in Bayesian optimization settings.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from botorch.posteriors.fully_bayesian import GaussianMixturePosterior

def parity_plot(truth:torch.Tensor, posterior:GaussianMixturePosterior,
                conf_int:bool=True, alpha:float=0.05, title:str="Parity Plot",
                **tkwargs:dict) -> None:
    """
    This function will create a parity plot which plots the ground truth against
    the predicted value from some model, with or without confidence intervals
    for each point.

    Args:
        truth (torch.Tensor):
            Ground truth values for each point.
        posterior (GaussianMixturePosterior):
            Posterior at given (test) points.
        conf_int (bool, default=True):
            Boolean denoting whether to add confidence intervals.
        alpha (float, default=0.05):
            Level of confidence to use for confidence intervals. For example,
            if `alpha=0.05` then (1-`alpha`)*100% = 95% confidence intervals
            will be shown.
        title (str, default="Parity Plot"):
            Main title for plot.
        **tkwargs (dict):
            Tensor keyword arguments.

    Returns:
        None -> will show parity plot.
    """

    # Extract Median and Bounds from Posterior
    median = posterior.quantile(value=torch.Tensor([0.5], **tkwargs))
    if conf_int:
        lcb = posterior.quantile(value=torch.Tensor([alpha/2], **tkwargs))
        ucb = posterior.quantile(value=torch.Tensor([1-alpha/2], **tkwargs))

    # Initialize Figure
    _, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Set Axis Limits, Labels
    min_val = np.min(torch.cat((truth.squeeze(), median.squeeze())).numpy())
    max_val = np.max(torch.cat((truth.squeeze(), median.squeeze())).numpy())

    ax.set_xlim([min_val if min_val <= 0 else 0, max_val])
    ax.set_ylim([min_val if min_val <= 0 else 0, max_val])
    ax.set_xlabel("Ground Truth", fontsize=18)
    ax.set_ylabel("Prediction", fontsize=18)

    # Plot "Matching" Line
    ax.plot([min_val if min_val <= 0 else 0, max_val],
            [min_val if min_val <= 0 else 0, max_val], "k--", lw=2)

    if conf_int:
        # Plot Points with Confidence Intervals
        yerr1, yerr2 = median - lcb, ucb - median
        ax.errorbar(
            truth.squeeze(-1).cpu().numpy(),
            median.squeeze(-1).cpu().numpy(),
            torch.cat((yerr1.unsqueeze(0), yerr2.unsqueeze(0)), 0).squeeze(-1),
            fmt=".",
            c="r",
            ecolor="gray",
            elinewidth=2,
            capsize=4
        )
    else:
        # Plot Points without Confidence Intervals
        ax.scatter(truth.squeeze(-1).cpu().numpy(),
                   median.squeeze(-1).cpu().numpy(),
                   marker=".", c="r")

    # Add Other Features to Plot
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=24)
    ax.grid(True)
    plt.show()


def progress_plot(Y:torch.Tensor, optimum:float=None, maximize:bool=True,
                  title:str="Optimization Progress") -> None:
    """
    This function will plot the progress across the number of evaluations
    (i.e., best value found). Note that in keeping with the BoTorch convention,
    we assume maximization but this can be controlled via `maximize`.

    Args:
        Y (torch.Tensor):
            Tensor containing evaluation values.
        optimum (float, default=None):
            Optimum that progress should converge towards.
        maximize (bool, default=True):
            Boolean indicating optimization focus (i.e., maximize or minimize).
        title (str, default="Optimization Progress"):
            Main title for plot.
    """

    # Initialize Figure
    _, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Extract Data
    Y = Y.cpu().numpy()
    n_eval = len(Y)
    best_values = np.minimum.accumulate(Y)

    # Plot Progress
    ax.plot(best_values, color="b", label="Progress")
    if optimum:
        if maximize:
            ax.plot([0, n_eval], [optimum, optimum], "k--", label="Maximum")
        else:
            ax.plot([0, n_eval], [optimum, optimum], "k--", label="Minimum")
    
    # Add Other Features
    ax.set_xlim([0, n_eval])
    ax.set_ylim([0, best_values[0]])
    ax.set_xlabel("Number of Evaluations", fontsize=18)
    ax.set_ylabel("Best Value Found", fontsize=18)
    ax.set_title(title, fontsize=24)
    ax.legend(fontsize=18)
    plt.show()
