from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jax_learning.equinox_basics import (
    filter_demo,
    filter_freeze_bias_demo,
    mlp_forward_demo,
    module_demo,
    module_parameter_count_demo,
    train_with_optax,
    train_with_optax_summary,
)


def _load_linear_data() -> tuple[jnp.ndarray, jnp.ndarray]:
    raw = np.loadtxt(Path("tests/fixtures/linear_regression_points.csv"), delimiter=",", skiprows=1)
    x = jnp.asarray(raw[:, 0:1], dtype=jnp.float32)
    y = jnp.asarray(raw[:, 1:2], dtype=jnp.float32)
    return x, y


def test_module_demo_runs_forward() -> None:
    x = jnp.array([[1.0], [2.0]], dtype=jnp.float32)
    out = module_demo(x)
    np.testing.assert_allclose(out, jnp.zeros_like(x))


def test_module_parameter_count_demo() -> None:
    assert module_parameter_count_demo() == 2


def test_filter_demo_splits_arrays_and_static_bits() -> None:
    arrays, static = filter_demo()
    assert hasattr(arrays, "weight")
    assert arrays.name is None
    assert static.name == "tiny"


def test_filter_freeze_bias_demo() -> None:
    trainable, static = filter_freeze_bias_demo()
    assert hasattr(trainable, "weight")
    assert trainable.bias is None
    assert static.bias == 0.5


def test_train_with_optax_learns_simple_line() -> None:
    x, y = _load_linear_data()
    model, losses = train_with_optax(x, y, steps=250, learning_rate=0.05)
    assert losses[-1] < losses[0]
    assert np.isclose(float(model.weight[0]), 2.0, atol=0.2)
    assert np.isclose(float(model.bias[0]), -1.0, atol=0.2)


def test_train_with_optax_summary() -> None:
    x, y = _load_linear_data()
    summary = train_with_optax_summary(x, y, steps=120, learning_rate=0.05)
    assert len(summary.losses) == 120
    assert summary.losses[-1] < summary.losses[0]


def test_mlp_forward_demo_has_expected_shape() -> None:
    out = mlp_forward_demo(seed=0)
    assert out.shape == (2, 1)
