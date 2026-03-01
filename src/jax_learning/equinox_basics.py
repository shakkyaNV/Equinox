"""Detailed Equinox learning examples: modules, filters, and Optax."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import optax


@dataclass(frozen=True)
class EquinoxTrainSummary:
    losses: list[float]
    weight: float
    bias: float


class TinyLinear(eqx.Module):
    """A single-neuron linear model."""

    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, *, weight: float = 0.0, bias: float = 0.0) -> None:
        self.weight = jnp.array([weight], dtype=jnp.float32)
        self.bias = jnp.array([bias], dtype=jnp.float32)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * self.weight + self.bias


class TinyMLP(eqx.Module):
    """A small MLP wrapper to show nested modules."""

    network: eqx.nn.MLP

    def __init__(self, key: jax.Array) -> None:
        self.network = eqx.nn.MLP(
            in_size=2,
            out_size=1,
            width_size=8,
            depth=2,
            key=key,
            activation=jax.nn.relu,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.network(x)


def module_demo(x: jnp.ndarray) -> jnp.ndarray:
    """Forward pass with a tiny linear Equinox module."""
    model = TinyLinear()
    return model(x)


def module_parameter_count_demo() -> int:
    """Count trainable array elements in TinyLinear."""
    model = TinyLinear()
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    return int(sum(leaf.size for leaf in leaves))


def filter_demo() -> tuple[eqx.Module, eqx.Module]:
    """Split trainable arrays from static fields."""

    class MixedModule(eqx.Module):
        weight: jnp.ndarray
        name: str

    mixed = MixedModule(weight=jnp.array([1.0], dtype=jnp.float32), name="tiny")
    arrays, static = eqx.partition(mixed, eqx.is_array)
    return arrays, static


def filter_freeze_bias_demo() -> tuple[eqx.Module, eqx.Module]:
    """A practical freezing idea: make bias static so only weights optimize."""

    class SemiFrozenLinear(eqx.Module):
        weight: jnp.ndarray
        bias: float

    model = SemiFrozenLinear(weight=jnp.array([1.0], dtype=jnp.float32), bias=0.5)
    trainable, static = eqx.partition(model, eqx.is_array)
    return trainable, static


def train_with_optax(
    x: jnp.ndarray,
    y: jnp.ndarray,
    steps: int = 150,
    learning_rate: float = 0.1,
) -> tuple[TinyLinear, list[float]]:
    """Minimal Equinox + Optax loop."""
    model = TinyLinear()
    optimizer = optax.sgd(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_value_and_grad
    def loss_fn(current_model: TinyLinear, features: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        preds = current_model(features)
        return jnp.mean((preds - targets) ** 2)

    @eqx.filter_jit
    def step(current_model: TinyLinear, state: optax.OptState) -> tuple[TinyLinear, optax.OptState, jnp.ndarray]:
        loss, grads = loss_fn(current_model, x, y)
        updates, new_state = optimizer.update(grads, state, current_model)
        new_model = eqx.apply_updates(current_model, updates)
        return new_model, new_state, loss

    losses: list[float] = []
    for _ in range(steps):
        model, opt_state, loss = step(model, opt_state)
        losses.append(float(loss))

    return model, losses


def train_with_optax_summary(
    x: jnp.ndarray,
    y: jnp.ndarray,
    steps: int = 200,
    learning_rate: float = 0.05,
) -> EquinoxTrainSummary:
    """Train and return compact metrics for quick notebook display."""
    model, losses = train_with_optax(x, y, steps=steps, learning_rate=learning_rate)
    return EquinoxTrainSummary(
        losses=losses,
        weight=float(model.weight[0]),
        bias=float(model.bias[0]),
    )


def mlp_forward_demo(seed: int = 0) -> jnp.ndarray:
    """Instantiate an MLP and run a batch through it."""
    model = TinyMLP(jax.random.PRNGKey(seed))
    batch = jnp.array([[1.0, 2.0], [-1.0, 0.5]], dtype=jnp.float32)
    return jax.vmap(model)(batch)
