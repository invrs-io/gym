"""Evaluate the solution to an invrs-gym challenge.

Example usage:

    python scripts/eval.py metagrating reference_designs/metagrating

Copyright (c) 2024 The INVRS-IO authors.
"""

import argparse
import glob
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as onp
from invrs_gym import challenges
from totypes import json_utils, types

# 64 bit mode ensures highest accuracy for evaluation results.
jax.config.update("jax_enable_x64", True)

PyTree = Any


def load_solutions(path: str) -> Dict[str, PyTree]:
    """Loads solutions in `csv` or `json` format."""

    if path.endswith(".json") or path.endswith(".csv"):
        paths = [path]
    else:
        paths = glob.glob(f"{path}/*.json") + glob.glob(f"{path}/*.csv")

    solutions = {}
    for path in paths:
        if path.endswith(".json"):
            with open(path) as f:
                serialized_solution = f.read()
            solutions[path] = json_utils.pytree_from_json(serialized_solution)
        elif path.endswith(".csv"):
            density_array = onp.genfromtxt(path, delimiter=",")
            solutions[path] = types.Density2DArray(
                array=density_array, lower_bound=0.0, upper_bound=1.0
            )

    # Ensure that there are no weak types.
    solutions = jax.tree_util.tree_map(lambda x: jnp.array(x, x.dtype), solutions)
    return solutions


def evaluate_solutions(challenge_name: str, path: str, output_path: str) -> None:
    """Evaluate solutions to the specified challenge."""
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("64-bit mode is required for eval calculations.")

    solutions = load_solutions(path)
    challenge = challenges.BY_NAME[challenge_name]()  # type: ignore[operator]

    with jax.default_device(jax.devices("cpu")[0]):

        @jax.jit
        def eval_metrics_fn(params):
            response, aux = challenge.component.response(params)
            metrics = challenge.metrics(response, params=params, aux=aux)
            return metrics

        for solution_path, solution in solutions.items():
            metrics = eval_metrics_fn(params=solution)
            metrics_str = " ".join(
                [
                    f"{metric_name}={float(metric_value)}"
                    for metric_name, metric_value in metrics.items()
                    if jnp.size(metric_value) == 1  # Scalar metrics only.
                ]
            )
            output_str = f"{challenge_name} {solution_path}: " + metrics_str
            print(output_str)
            if bool(output_path):
                with open(output_path, "a") as f:
                    f.write(output_str)
                    f.write("\n")


# -----------------------------------------------------------------------------
# Command line interface.
# -----------------------------------------------------------------------------


parser = argparse.ArgumentParser(
    prog="eval",
    description="Evaluate the solution to an invrs-gym challenge",
)
parser.add_argument(
    "challenge",
    type=str,
    choices=list(challenges.BY_NAME.keys()),
    help="The name of the invrs-gym challenge",
)
parser.add_argument(
    "path",
    type=str,
    help="Path to solution, or directory containing multiple solutions",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="",
    required=False,
    help="Optional output path, with `txt` extension",
)


if __name__ == "__main__":
    args = parser.parse_args()
    evaluate_solutions(
        challenge_name=args.challenge,
        path=args.path,
        output_path=args.output_path,
    )
