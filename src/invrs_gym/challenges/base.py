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
    def init(self, key: jax.Array):
        raise NotImplementedError

    def response(self, params: PyTree) -> Tuple[Any, AuxDict]:
        raise NotImplementedError


class Challenge(abc.ABC):
    component: Component

    def loss(self, response: Any) -> jnp.ndarray:
        raise NotImplementedError

    def metrics(self, response: Any, params: PyTree, aux: AuxDict) -> AuxDict:
        raise NotImplementedError
