"""Base types for challenges and components.

Copyright (c) 2023 The INVRS-IO authors.
"""

import abc
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
from totypes import types

AuxDict = Dict[str, Any]
PyTree = Any


DensityInitializer = Callable[[jax.Array, types.Density2DArray], types.Density2DArray]


class Component(abc.ABC):
    """Base class for components.

    The component represents the physical structure to be optimized. Initial parameters
    are returned by the `init` method; these can include types having metadata that
    specifies bounds on parameter values or target characteristics, e.g. a minimum
    feature size.

    The component has a default "excitation", e.g. illumination with a particular
    wavelength of light from a specific direction; the response of the component to the
    excitation is obtained with the `response` method. The `response` method of classes
    inheriting from `Component` will generally have optional arguments to modify the
    excitation condition.
    """

    def init(self, key: jax.Array):
        """Returns the initial parameters given a `PRNGKey`."""
        raise NotImplementedError

    def response(self, params: PyTree) -> Tuple[Any, AuxDict]:
        """Returns the response of the component for the given `params`."""
        raise NotImplementedError


class Challenge(abc.ABC):
    """Base class for challenges.

    Challenges consist of a `Component` to be optimized and a loss function which
    returns a scalar loss given the component response.

    Attributes:
        component: The `Component` to be optimized.
    """

    component: Component

    def loss(self, response: Any) -> jnp.ndarray:
        """Compute scalar loss for the `response`."""
        raise NotImplementedError

    def metrics(self, response: Any, params: PyTree, aux: AuxDict) -> AuxDict:
        """Compute metrics for a component response and associated quantities."""
        raise NotImplementedError
