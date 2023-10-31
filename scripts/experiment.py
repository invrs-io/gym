"""Runs an experiment.

Example usage:

    python experiment.py --path="experiments/" --steps=100

Note that on some machines, use of more than one worker can be unexpectedly slow.

Copyright (c) 2023 The INVRS-IO authors.
"""

import argparse
import dataclasses
import glob
import itertools
import json
import multiprocessing as mp
import os
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

Sweep = List[Dict[str, Any]]


def run_experiment(
    experiment_path: str,
    workers: int,
    dry_run: bool,
    randomize: bool,
    steps: int,
) -> None:
    """Runs an experiment."""

    # Define the experiment.
    challenge_sweeps = sweep("challenge_name", ["metagrating"])
    hparam_sweeps = sweep_product(
        sweep("density_mean_value", [0.5]),
        sweep("density_noise_stddev", [0.1]),
        sweep("beta", [2.0]),
        sweep("seed", range(3)),
        sweep("steps", [steps]),
    )
    sweeps = sweep_product(challenge_sweeps, hparam_sweeps)

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
    return run_work_unit(wid_path=wid_path, **kwargs)


def sweep(name: str, values: Sequence[Any]) -> Sweep:
    """Generate a list of dictionaries defining a sweep."""
    return [{name: v} for v in values]


def sweep_zip(*sweeps: Sweep) -> Sweep:
    """Zip sweeps of different variables."""
    return [_merge(*kw) for kw in zip(*sweeps, strict=True)]


def sweep_product(*sweeps: Sweep) -> Sweep:
    """Return the Cartesian product of multiple sweeps."""
    return [_merge(*kw) for kw in itertools.product(*sweeps)]


def _merge(*vars: Dict[str, Any]) -> Dict[str, Any]:
    """Merge dictionaries defining sweeps of multiple variables."""
    d = {}
    for v in vars:
        d.update(v)
    return d


# -----------------------------------------------------------------------------
# Functions related to individual work units within the experiment.
# -----------------------------------------------------------------------------


def run_work_unit(
    wid_path: str,
    challenge_name: str,
    steps: int,
    seed: int = 0,
    beta: float = 2.0,
    density_mean_value: float = 0.5,
    density_noise_stddev: float = 0.1,
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
    import time

    import invrs_opt
    import jax
    from jax import numpy as jnp
    from totypes import json_utils, types

    from invrs_gym import challenges
    from invrs_gym.utils import initializers

    # Create a basic checkpoint manager that can serialize custom types.
    mngr = CheckpointManager(
        path=wid_path,
        save_interval_steps=10,
        max_to_keep=1,
        serialize_fn=json_utils.json_from_pytree,
        deserialize_fn=json_utils.pytree_from_json,
    )

    # Define a custom density initializer that returns a density with the prescribed
    # initial value with added random noise.
    def density_initializer(
        key: jax.Array,
        seed_density: types.Density2DArray,
    ) -> types.Density2DArray:
        seed_density = dataclasses.replace(
            seed_density,
            array=jnp.full(seed_density.shape, density_mean_value),
        )
        return initializers.noisy_density_initializer(
            key, density=seed_density, relative_stddev=density_noise_stddev
        )

    challenge_kwargs.update({"density_initializer": density_initializer})
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
        checkpoint = mngr.restore(latest_step)
        state = checkpoint["state"]
        scalars = checkpoint["scalars"]
    else:
        latest_step = -1  # Next step is `0`.
        params = challenge.component.init(jax.random.PRNGKey(seed))
        state = opt.init(params)
        scalars = {}

    def _log_scalar(name: str, value: float) -> None:
        if name not in scalars:
            scalars[name] = jnp.zeros((0,))
        scalars[name] = jnp.concatenate([scalars[name], jnp.asarray([float(value)])])

    for i in range(latest_step + 1, steps):
        params = opt.params(state)
        t0 = time.time()
        (loss_value, (_, distance, metrics, _)), grad = value_and_grad_fn(params)
        t1 = time.time()

        _log_scalar("loss", loss_value)
        _log_scalar("distance", distance)
        _log_scalar("simulation_time", t1 - t0)
        _log_scalar("update_time", time.time() - t0)
        for key, metric_value in metrics.items():
            if _is_scalar(metric_value):
                _log_scalar(key, metric_value)
        mngr.save(i, {"state": state, "scalars": scalars, "params": params})

        if stop_on_zero_distance and distance <= 0:
            break
        state = opt.update(value=loss_value, params=params, grad=grad, state=state)

    mngr.save(
        i, {"state": state, "scalars": scalars, "params": params}, force_save=True
    )
    with open(wid_path + "/completed.txt", "w") as f:
        os.utime(wid_path, None)


# -----------------------------------------------------------------------------
# Functions related to checkpointing.
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class CheckpointManager:
    """A simple checkpoint manager with an orbax-like API."""

    path: str
    save_interval_steps: int
    max_to_keep: int
    serialize_fn: Callable[[Any], str] = json.dumps
    deserialize_fn: Callable[[str], Any] = json.loads

    def latest_step(self) -> Optional[int]:
        """Return the latest checkpointed step, or `None` if no checkpoints exist."""
        steps = self._checkpoint_steps()
        steps.sort()
        return None if len(steps) == 0 else steps[-1]

    def save(self, step: int, pytree: Any, force_save: bool = False) -> None:
        """Save a pytree checkpoint."""
        if (step + 1) % self.save_interval_steps != 0 and not force_save:
            return
        with open(self._checkpoint_fname(step), "w") as f:
            f.write(self.serialize_fn(pytree))
        steps = self._checkpoint_steps()
        steps.sort()
        steps_to_delete = steps[: -self.max_to_keep]
        for step in steps_to_delete:
            os.remove(self._checkpoint_fname(step))

    def restore(self, step: int) -> Any:
        """Restore a pytree checkpoint."""
        with open(self._checkpoint_fname(step)) as f:
            return self.deserialize_fn(f.read())

    def _checkpoint_steps(self) -> List[int]:
        """Return the steps for which checkpoint files exist."""
        fnames = glob.glob(self.path + "/checkpoint_*.json")
        return [int(f.split("_")[-1][:-5]) for f in fnames]

    def _checkpoint_fname(self, step: int) -> str:
        """Return the chackpoint filename for the given step."""
        return self.path + f"/checkpoint_{step:04}.json"


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
