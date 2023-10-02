"""Implements a basic optimization."""


from typing import Any, Callable, Dict, Protocol, Tuple

import jax
import jax.numpy as jnp
import optax  # type: ignore[import]
from jax import tree_util
from totypes import types  # type: ignore[import]

AuxDict = Dict[str, Any]


class Component(Protocol):
    def init(self, key: jax.Array) -> Any:
        ...

    def response(self, params: Any) -> Tuple[Any, AuxDict]:
        ...


class Challenge(Protocol):
    component: Component

    def loss(self, response: Any) -> jnp.ndarray:
        ...

    def metrics(self, response: Any, params: Any, aux: AuxDict) -> AuxDict:
        ...


def setup_optimization(
    challenge: Challenge,
    optimizer: optax.GradientTransformation,
    response_kwargs: Dict[str, Any] = {},
) -> Tuple[
    Any,  # params
    Any,  # state
    # f(params, state) -> params, state, (value, response, aux, metrics)
    Callable[
        [Any, Any],
        Tuple[Any, Any, Tuple[jnp.ndarray, Any, AuxDict, AuxDict]],
    ],
]:
    def loss_fn(params: Any) -> Tuple[jnp.ndarray, Tuple[Any, AuxDict]]:
        response, aux = challenge.component.response(params, **response_kwargs)
        loss = challenge.loss(response)
        return loss, (response, aux)

    def clip(
        leaf: types.BoundedArray | types.Density2DArray,
    ) -> types.BoundedArray | types.Density2DArray:
        (value,), treedef = tree_util.tree_flatten(leaf)
        return tree_util.tree_unflatten(
            treedef, (jnp.clip(value, leaf.lower_bound, leaf.upper_bound),)
        )

    def apply_fixed_pixels(
        leaf: types.Density2DArray,
    ) -> types.Density2DArray:
        (value,), treedef = tree_util.tree_flatten(leaf)
        if leaf.fixed_solid is not None:
            value = jnp.where(leaf.fixed_solid, leaf.upper_bound, value)
        if leaf.fixed_void is not None:
            value = jnp.where(leaf.fixed_void, leaf.lower_bound, value)
        return tree_util.tree_unflatten(treedef, (value,))

    def transform(leaf: Any) -> Any:
        if isinstance(leaf, types.BoundedArray):
            return clip(leaf)
        if isinstance(leaf, types.Density2DArray):
            leaf = types.symmetrize_density(leaf)
            leaf = clip(leaf)
            return apply_fixed_pixels(leaf)
        return leaf

    def step_fn(
        params: Any, state: Any
    ) -> Tuple[Any, Any, Tuple[jnp.ndarray, Any, AuxDict, AuxDict]]:
        (value, (response, aux)), grad = jax.value_and_grad(loss_fn, has_aux=True)(
            params
        )
        metrics = challenge.metrics(response, params, aux)
        updates, state = optimizer.update(grad, state)
        params = optax.apply_updates(params, updates)
        params = tree_util.tree_map(
            transform, params, is_leaf=lambda x: isinstance(x, types.CUSTOM_TYPES)
        )
        return params, state, (value, response, aux, metrics)

    params = challenge.component.init(jax.random.PRNGKey(0))

    # Eliminate weak types by specifying types of all arrays in the params pytree.
    params = tree_util.tree_map(lambda x: jnp.asarray(x, jnp.asarray(x).dtype), params)
    state = optimizer.init(params)
    return params, state, step_fn
