"""Runs an experiment.

Example usage:

    python scripts/experiment.py --path="experiments/test" --steps=100

Note that on some machines, use of more than one worker can be unexpectedly slow.

Copyright (c) 2023 The INVRS-IO authors.
"""

import argparse
import functools
import json
import multiprocessing as mp
import os
import random
import time
from typing import Any, Dict, List, Tuple

from invrs_utils.experiment import sweep

Sweep = List[Dict[str, Any]]

PRINT_INTERVAL = 300


def run_experiment(
    experiment_path: str,
    workers: int,
    dry_run: bool,
    randomize: bool,
    steps: int,
) -> None:
    """Runs an experiment."""

    # Define the experiment.
    challenge_sweeps = sweep.sweep("challenge_name", ["metagrating"])
    hparam_sweeps = sweep.product(
        sweep.sweep("density_relative_mean", [0.5]),
        sweep.sweep("density_relative_noise_amplitude", [0.1]),
        sweep.sweep("beta", [2.0]),
        sweep.sweep("seed", range(3)),
        sweep.sweep("steps", [steps]),
    )
    sweeps = sweep.product(challenge_sweeps, hparam_sweeps)

    # Set up checkpointing directory.
    wid_paths = [experiment_path + f"/wid_{i:04}" for i in range(len(sweeps))]

    # Print some information about the experiment.
    print(
        f"Experiment:\n"
        f"  worker count = {max(1, workers)}\n"
        f"  work unit count = {len(sweeps)}\n"
        f"  experiment path = {experiment_path}\n"
        f"Work units:"
    )
    for wid_path, kwargs in zip(wid_paths, sweeps):
        print(f"  {wid_path}: {kwargs}")

    path_and_kwargs = list(zip(wid_paths, sweeps))
    if randomize:
        random.shuffle(path_and_kwargs)

    if dry_run:
        return

    with mp.Pool(processes=workers) as pool:
        _ = list(pool.imap_unordered(_run_work_unit, path_and_kwargs))


def _run_work_unit(path_and_kwargs: Tuple[str, Dict[str, Any]]) -> None:
    """Wraps `run_work_unit` so that it can be called by `map`."""
    wid_path, kwargs = path_and_kwargs
    run_work_unit(wid_path, **kwargs)


# -----------------------------------------------------------------------------
# Functions related to individual work units within the experiment.
# -----------------------------------------------------------------------------


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

    print(f"{wid_path} starting")
    last_print_time = time.time()

    # The use of multiprocessing requires that some modules be imported here, as they
    # cannot be imported in the main process which is forked.
    import invrs_opt
    import jax
    from invrs_utils.experiment import checkpoint
    from jax import numpy as jnp

    from invrs_gym import challenges
    from invrs_gym.utils import initializers

    # Create a basic checkpoint manager that can serialize custom types.
    mngr = checkpoint.CheckpointManager(
        path=wid_path,
        save_interval_steps=10,
        max_to_keep=1,
    )

    challenge_kwargs.update(
        {
            "density_initializer": functools.partial(
                initializers.noisy_density_initializer,
                relative_mean=density_relative_mean,
                relative_noise_amplitude=density_relative_noise_amplitude,
            ),
        }
    )
    challenge = challenges.BY_NAME[challenge_name](  # type: ignore[operator]
        **challenge_kwargs
    )

    def loss_fn(
        params: Any,
    ) -> Tuple[jnp.ndarray, Tuple[Any, jnp.ndarray, Dict[str, Any], Dict[str, Any]]]:
        response, aux = challenge.component.response(params)
        loss = challenge.loss(response)
        distance = challenge.distance_to_target(response)
        metrics = challenge.metrics(response, params, aux)
        return loss, (response, distance, metrics, aux)

    # Use a jit-compiled value-and-grad function, if the challenge supports it.
    value_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    try:
        dummy_params = challenge.component.init(jax.random.PRNGKey(0))
        value_and_grad_fn = jax.jit(value_and_grad_fn).lower(dummy_params).compile()
    except jax.errors.TracerArrayConversionError:
        pass

    opt = invrs_opt.density_lbfgsb(beta=beta)
    if mngr.latest_step() is not None:
        latest_step: int = mngr.latest_step()  # type: ignore[assignment]
        latest_checkpoint = mngr.restore(latest_step)
        state = latest_checkpoint["state"]
        scalars = latest_checkpoint["scalars"]
        min_loss = latest_checkpoint["min_loss"]
        champion_result = latest_checkpoint["champion_result"]
    else:
        latest_step = -1  # Next step is `0`.
        params = challenge.component.init(jax.random.PRNGKey(seed))
        state = opt.init(params)
        scalars = {}
        min_loss = jnp.inf
        champion_result = {}

    def _log_scalar(name: str, value: float) -> None:
        if name not in scalars:
            scalars[name] = jnp.zeros((0,))
        scalars[name] = jnp.concatenate([scalars[name], jnp.asarray([float(value)])])

    for i in range(latest_step + 1, steps):
        params = opt.params(state)
        t0 = time.time()
        (loss_value, (response, distance, metrics, aux)), grad = value_and_grad_fn(
            params
        )
        t1 = time.time()

        if time.time() > last_print_time + PRINT_INTERVAL:
            last_print_time = time.time()
            print(f"{wid_path} is now at step {i}")

        _log_scalar("loss", loss_value)
        _log_scalar("distance", distance)
        _log_scalar("simulation_time", t1 - t0)
        _log_scalar("update_time", time.time() - t0)
        for key, metric_value in metrics.items():
            if _is_scalar(metric_value):
                _log_scalar(key, metric_value)
        if i == 0 or loss_value < min_loss:
            min_loss = loss_value
            champion_result = {
                "params": params,
                "loss": loss_value,
                "response": response,
                "distance": distance,
                "metrics": metrics,
                "aux": aux,
            }
        ckpt_dict = {
            "state": state,
            "scalars": scalars,
            "params": params,
            "min_loss": min_loss,
            "champion_result": champion_result,
        }
        mngr.save(i, ckpt_dict)
        if stop_on_zero_distance and distance <= 0:
            break
        state = opt.update(value=loss_value, params=params, grad=grad, state=state)

    mngr.save(i, ckpt_dict, force_save=True)
    with open(wid_path + "/completed.txt", "w") as f:
        os.utime(wid_path, None)

    print(f"{wid_path} finished")


def _is_scalar(x: Any) -> bool:
    """Returns `True` if `x` is a scalar, i.e. it can be cast as a float."""
    try:
        float(x)
        return True
    except Exception:
        return False


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
    run_experiment(
        experiment_path=args.path,
        workers=args.workers,
        dry_run=args.dry_run,
        randomize=args.randomize,
        steps=args.steps,
    )
