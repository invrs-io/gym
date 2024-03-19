"""Runs an experiment.

Example usage:

    python scripts/experiment.py --path="experiments/test" --steps=100

Note that on some machines, use of more than one worker can be unexpectedly slow.

Copyright (c) 2023 The INVRS-IO authors.
"""

import argparse
import functools
import json
import os
from typing import Any, Dict, Sequence

from invrs_utils.experiment import experiment, sweep

PRINT_INTERVAL = 300


def experiment_sweeps(steps: int) -> Sequence[Dict[str, Any]]:
    """Defines the hyperparameter sweep of the experiment."""
    challenge_sweeps = sweep.sweep("challenge_name", ["metagrating"])
    hparam_sweeps = sweep.product(
        sweep.sweep("density_relative_mean", [0.5]),
        sweep.sweep("density_relative_noise_amplitude", [0.1]),
        sweep.sweep("beta", [2.0]),
        sweep.sweep("seed", range(3)),
        sweep.sweep("steps", [steps]),
    )
    return sweep.product(challenge_sweeps, hparam_sweeps)


@experiment.work_unit_fn
def run_work_unit(
    wid_path: str,
    challenge_name: str,
    steps: int,
    seed: int = 0,
    beta: float = 2.0,
    density_relative_mean: float = 0.5,
    density_relative_noise_amplitude: float = 0.1,
    stop_on_zero_distance: bool = True,
    **challenge_kwargs: Any,
) -> None:
    """Runs a work unit."""
    if os.path.isfile(wid_path + "/completed.txt"):
        return
    if not os.path.exists(wid_path):
        os.makedirs(wid_path)

    # Save the work unit configuration to the checkpoint directory.
    work_unit_config = locals()
    with open(wid_path + "/setup.json", "w") as f:
        json.dump(work_unit_config, f, indent=4)

    # The use of multiprocessing requires that some modules be imported here, as they
    # cannot be imported in the main process which is forked.
    import invrs_opt
    from invrs_utils.experiment import work_unit
    from jax import random

    from invrs_gym import challenges
    from invrs_gym.utils import initializers

    challenge = challenges.BY_NAME[challenge_name](  # type: ignore[operator]
        density_initializer=functools.partial(
            initializers.noisy_density_initializer,
            relative_mean=density_relative_mean,
            relative_noise_amplitude=density_relative_noise_amplitude,
        ),
        **challenge_kwargs,
    )

    work_unit.run_work_unit(
        key=random.PRNGKey(seed),
        wid_path=wid_path,
        challenge=challenge,
        optimizer=invrs_opt.density_lbfgsb(beta=beta),
        steps=steps,
        stop_on_zero_distance=stop_on_zero_distance,
        stop_requires_binary=True,
        save_interval_steps=10,
        max_to_keep=1,
        print_interval=60,
    )


# -----------------------------------------------------------------------------
# Command line interface.
# -----------------------------------------------------------------------------


parser = argparse.ArgumentParser(
    prog="experiment",
    description="Run an experiment with multiple work units",
)
parser.add_argument(
    "--workers",
    type=int,
    default=1,
    help="Number of work units to run in parallel",
)
parser.add_argument(
    "--steps",
    type=int,
    default=200,
    help="Maximum number of optimization steps",
)
parser.add_argument(
    "--path",
    type=str,
    default="",
    help="Relative experiment path",
)
parser.add_argument(
    "--dry-run",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Configures but does not launch experiment",
)
parser.add_argument(
    "--randomize",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Randomizes the order in which work units are executed",
)

if __name__ == "__main__":
    args = parser.parse_args()
    experiment.run_experiment(
        experiment_path=args.path,
        sweeps=experiment_sweeps(steps=args.steps),
        work_unit_fn=run_work_unit,
        workers=args.workers,
        dry_run=args.dry_run,
        randomize=args.randomize,
    )
