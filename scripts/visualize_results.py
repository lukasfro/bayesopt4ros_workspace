import argparse
import json
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

from botorch.models import SingleTaskGP
import torch


def show_convergence(log_dir, axis):
    eval_dict = load_eval(log_dir)

    num_evals = eval_dict["train_targets"].shape[0]
    eval_iter = np.arange(num_evals) + 1

    axis.plot(eval_iter, eval_dict["y_best"], "C3", label="Best function value")
    axis.plot(eval_iter, eval_dict["train_targets"], "ko", label="Function value")
    axis.set_xlabel("# Iteration")
    axis.set_ylabel("Function value")
    axis.legend()


def show_model(log_dir, axis):
    # Load data from files
    gp = load_gp(log_dir)
    config = load_config(log_dir)
    eval_dict = load_eval(log_dir)

    # Visualization depends on dimensionality of the input space
    assert config["input_dim"] == 1

    lb, ub = config["lower_bound"][0], config["upper_bound"][0]
    x_plot = torch.linspace(lb, ub, 500)
    posterior = gp.posterior(x_plot.unsqueeze(-1))
    mean = posterior.mean.squeeze().detach()
    upper, lower = posterior.mvn.confidence_region()

    # mean, var = gp.posterior.mean(x_plot)
    # mean = mean.squeeze()
    # std = np.sqrt(var).squeeze()

    axis.plot(x_plot, mean, label="GP mean")
    axis.fill_between(x_plot, upper.detach(), lower.detach(), label="95% confidence", alpha=0.3)
    axis.plot(eval_dict["train_inputs"], eval_dict["train_targets"], "ko", label="Data")
    axis.plot(eval_dict["x_best"][-1], eval_dict["y_best"][-1], "C8*", label="Best point")
    axis.set_xlabel("Optimzation parameter")
    axis.set_ylabel("Function value")
    axis.legend()


def load_gp(log_dir):
    try:
        # The GP state, i.e., hyperparameters, normalization, etc.
        model_file = os.path.join(log_dir, "model_state.pth")
        with open(model_file, "rb") as f:
            state_dict = torch.load(f)

        # Get the evaluated data points
        eval_dict = load_eval(log_dir)
        train_X = eval_dict["train_inputs"]
        train_Y = eval_dict["train_targets"]

        # The bounds of the domain
        config_dict = load_config(log_dir)
        lb = torch.tensor(config_dict["lower_bound"])
        ub = torch.tensor(config_dict["upper_bound"])
        bounds = torch.stack((lb, ub))

        # Create GP instance and load respective parameters
        gp = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            outcome_transform=Standardize(m=1),
            input_transform=Normalize(d=1, bounds=bounds),
        )
        gp.load_state_dict(state_dict=state_dict)
    except FileNotFoundError:
        print(f"The model file could not be found in: {log_dir}")
        exit(1)
    return gp


def load_config(log_dir):
    try:
        config_file = os.path.join(log_dir, "config.yaml")
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print(f"The config file could not be found in: {log_dir}")
        exit(1)
    return config


def load_eval(log_dir):
    try:
        evaluations_file = os.path.join(log_dir, "evaluations.yaml")
        with open(evaluations_file, "r") as f:
            eval_dict = yaml.load(f, Loader=yaml.FullLoader)
        eval_dict = {k: torch.tensor(v) for k, v in eval_dict.items()}
    except FileNotFoundError:
        print(f"The evaluations file could not be found in: {log_dir}")
        exit(1)
    return eval_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--log-dir",
        help="Logging directory with the results to plot",
        type=str,
        default="./logs/",
    )

    args = parser.parse_args()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    fig.suptitle(f"BayesOpt4ROS \n Logging directory: {args.log_dir}")
    gp = load_gp(args.log_dir)

    show_convergence(args.log_dir, axes[0])
    show_model(args.log_dir, axes[1])
    plt.tight_layout()
    plt.savefig(os.path.join(args.log_dir, "final_visualization.png"))
    # plt.show()