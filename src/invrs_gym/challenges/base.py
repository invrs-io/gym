"""Base types for challenges and components.

Copyright (c) 2023 The INVRS-IO authors.
"""

import abc
from typing import Any, Callable, Dict, Tuple

import fmmax
import jax
import jax.numpy as jnp
from totypes import json_utils, types

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

    @abc.abstractmethod
    def init(self, key: jax.Array):
        """Returns the initial parameters given a `PRNGKey`."""

    @abc.abstractmethod
    def response(self, params: PyTree) -> Tuple[Any, AuxDict]:
        """Returns the response of the component for the given `params`."""


class Challenge(abc.ABC):
    """Base class for challenges.

    Challenges consist of a `Component` to be optimized and a loss function which
    returns a scalar loss given the component response.

    Challenges also include a `distance_to_target` function which returns a scalar
    that is zero or negative when the challenge objective is achieved. Unlike the
    loss function, `distance_to_target` may not be well-suited for gradient-based
    optimization and may not even be differentiable. The distance serves as an
    independent way to assess the quality of a solution, e.g. in cases where various
    loss functions are compared.

    Challenges also include a `metrics` function which computes additional quantities
    that may be useful assessing the quality of a particular component.

    Attributes:
        component: The `Component` to be optimized.
    """

    component: Component

    @abc.abstractmethod
    def loss(self, response: Any) -> jnp.ndarray:
        """Compute scalar loss for the `response`."""

    @abc.abstractmethod
    def distance_to_target(self, response: Any) -> jnp.ndarray:
        """Compute the distance between the response and the challenge target."""

    @abc.abstractmethod
    def metrics(self, response: Any, params: PyTree, aux: AuxDict) -> AuxDict:
        """Compute metrics for a component response and associated quantities."""


# Several challenges use the `fmmax` simulator, and contain `fmmax` custom objects in
# their responses. Ensure these are serializable by registering with totypes.
json_utils.register_custom_type(fmmax.basis.Expansion)
