"""Detailed JAX lesson snippets with tiny experiments.

These are intentionally small and practical. The idea is that each function can
be read in isolation and run quickly while learning.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class BasicsReport:
    """Runtime facts that are handy while getting started."""

    device_count: int
    default_backend: str
    devices: tuple[str, ...]


@dataclass(frozen=True)
class TrainMetrics:
    """Simple container for training/validation traces."""

    train_loss: list[float]
    val_loss: list[float]


def setup_and_devices() -> BasicsReport:
    """Collect JAX runtime details.

    Returns backend, device count, and each device string so it's obvious what
    hardware JAX sees.
    """
    visible_devices = tuple(str(device) for device in jax.devices())
    return BasicsReport(
        device_count=jax.device_count(),
        default_backend=jax.default_backend(),
        devices=visible_devices,
    )


def numpy_but_jax_demo() -> jnp.ndarray:
    """Show NumPy-like array creation, reshape, broadcasting, and reduction."""
    x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    centered = x - x.mean(axis=0, keepdims=True)
    scaled = centered * jnp.array([1.0, 0.5, 2.0, -1.0], dtype=jnp.float32)
    return scaled.mean(axis=1)


def numpy_indexing_and_where_demo() -> jnp.ndarray:
    """Mask and transform only positive values using `where`."""
    sample = jnp.array([-2.0, -1.0, 0.0, 1.5, 3.0], dtype=jnp.float32)
    return jnp.where(sample > 0.0, jnp.log1p(sample), 0.0)


def autodiff_grad_demo(x: float) -> float:
    """Differentiate x^2 + 3x + 2 at a point."""

    def poly(t: float) -> float:
        return t**2 + 3.0 * t + 2.0

    return float(jax.grad(poly)(x))


def autodiff_value_and_grad_demo(x: float) -> tuple[float, float]:
    """Return both function value and gradient for a smooth objective."""

    def objective(t: float) -> float:
        return jnp.sin(t) + 0.1 * t**2

    value, grad = jax.value_and_grad(objective)(x)
    return float(value), float(grad)


def higher_order_grad_demo(x: float) -> float:
    """Second derivative example: d²/dx²(x^4) = 12x²."""

    def quartic(t: float) -> float:
        return t**4

    second_derivative = jax.grad(jax.grad(quartic))
    return float(second_derivative(x))


def jit_demo(x: jnp.ndarray) -> jnp.ndarray:
    """JIT compile a small non-linear transform."""

    @jax.jit
    def transform(v: jnp.ndarray) -> jnp.ndarray:
        return jnp.tanh(v) + 0.25 * v

    return transform(x)


def vmap_demo(xs: jnp.ndarray) -> jnp.ndarray:
    """Batch scalar function evaluation with `vmap`."""

    def scalar_square_plus_one(v: jnp.ndarray) -> jnp.ndarray:
        return v**2 + 1.0

    return jax.vmap(scalar_square_plus_one)(xs)


def vmap_pairwise_l2_demo(points: jnp.ndarray, reference: jnp.ndarray) -> jnp.ndarray:
    """Compute L2 distance from many points to one reference vector."""

    def distance_fn(point: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.norm(point - reference)

    return jax.vmap(distance_fn)(points)


def pmap_demo(xs: jnp.ndarray) -> jnp.ndarray:
    """Run one shard per available device and add one."""

    device_count = jax.device_count()
    trimmed = xs[:device_count]

    @jax.pmap
    def add_one(v: jnp.ndarray) -> jnp.ndarray:
        return v + 1.0

    return add_one(trimmed)


def pmap_center_demo(xs: jnp.ndarray) -> jnp.ndarray:
    """A tiny `pmap` experiment: center each shard around its mean."""

    device_count = jax.device_count()
    trimmed = xs[:device_count]

    @jax.pmap
    def center(v: jnp.ndarray) -> jnp.ndarray:
        return v - jnp.mean(v)

    return center(trimmed)


def prng_demo(seed: int = 0) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Split PRNG key into independent streams for reproducibility."""
    key = jax.random.PRNGKey(seed)
    key_a, key_b = jax.random.split(key)
    normal = jax.random.normal(key_a, shape=(3,))
    uniform = jax.random.uniform(key_b, shape=(3,), minval=-1.0, maxval=1.0)
    return normal, uniform


def prng_dropout_mask_demo(seed: int, shape: tuple[int, ...], drop_rate: float = 0.2) -> jnp.ndarray:
    """Generate a boolean dropout mask with explicit randomness."""
    key = jax.random.PRNGKey(seed)
    keep_prob = 1.0 - drop_rate
    return jax.random.bernoulli(key, p=keep_prob, shape=shape)


def pure_function_step(weights: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, lr: float) -> jnp.ndarray:
    """One pure gradient-descent update for linear regression weights."""

    def loss_fn(w: jnp.ndarray) -> jnp.ndarray:
        preds = x @ w
        return jnp.mean((preds - y) ** 2)

    grads = jax.grad(loss_fn)(weights)
    return weights - lr * grads


def pytree_demo() -> dict[str, dict[str, jnp.ndarray]]:
    """Map over nested model-like parameters."""
    params = {
        "layer1": {"w": jnp.array([1.0, -1.0]), "b": jnp.array([0.5])},
        "layer2": {"w": jnp.array([2.0]), "b": jnp.array([-0.5])},
    }
    return jax.tree_util.tree_map(lambda leaf: leaf * 2.0, params)


def pytree_flatten_demo() -> tuple[int, tuple[int, ...]]:
    """Flatten a pytree and return leaf count + per-leaf sizes."""
    tree = {
        "encoder": [jnp.ones((2, 2), dtype=jnp.float32), jnp.zeros((2,), dtype=jnp.float32)],
        "decoder": (jnp.array([3.0], dtype=jnp.float32),),
    }
    leaves, _ = jax.tree_util.tree_flatten(tree)
    return len(leaves), tuple(int(leaf.size) for leaf in leaves)


def training_loop_demo(
    x: jnp.ndarray,
    y: jnp.ndarray,
    steps: int = 200,
    lr: float = 0.1,
) -> tuple[jnp.ndarray, list[float]]:
    """Tiny hand-rolled training loop with decreasing MSE history."""
    weights = jnp.zeros((x.shape[1],), dtype=jnp.float32)
    history: list[float] = []

    for _ in range(steps):
        weights = pure_function_step(weights, x, y, lr)
        preds = x @ weights
        history.append(float(jnp.mean((preds - y) ** 2)))

    return weights, history


def training_loop_with_validation_demo(
    train_x: jnp.ndarray,
    train_y: jnp.ndarray,
    val_x: jnp.ndarray,
    val_y: jnp.ndarray,
    steps: int = 250,
    lr: float = 0.05,
) -> tuple[jnp.ndarray, TrainMetrics]:
    """Train a line and track train + validation losses."""
    weights = jnp.zeros((train_x.shape[1],), dtype=jnp.float32)
    train_loss: list[float] = []
    val_loss: list[float] = []

    for _ in range(steps):
        weights = pure_function_step(weights, train_x, train_y, lr)
        train_preds = train_x @ weights
        val_preds = val_x @ weights
        train_loss.append(float(jnp.mean((train_preds - train_y) ** 2)))
        val_loss.append(float(jnp.mean((val_preds - val_y) ** 2)))

    return weights, TrainMetrics(train_loss=train_loss, val_loss=val_loss)
