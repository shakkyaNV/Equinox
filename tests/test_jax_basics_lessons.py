from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_learning.basics import (
    autodiff_grad_demo,
    autodiff_value_and_grad_demo,
    higher_order_grad_demo,
    jit_demo,
    numpy_but_jax_demo,
    numpy_indexing_and_where_demo,
    pmap_center_demo,
    pmap_demo,
    prng_demo,
    prng_dropout_mask_demo,
    pure_function_step,
    pytree_demo,
    pytree_flatten_demo,
    setup_and_devices,
    training_loop_demo,
    training_loop_with_validation_demo,
    vmap_demo,
    vmap_pairwise_l2_demo,
)


def _load_linear_data() -> tuple[jnp.ndarray, jnp.ndarray]:
    raw = np.loadtxt(Path("tests/fixtures/linear_regression_points.csv"), delimiter=",", skiprows=1)
    x = jnp.asarray(raw[:, 0:1], dtype=jnp.float32)
    y = jnp.asarray(raw[:, 1], dtype=jnp.float32)
    return x, y


def _load_wave_data() -> tuple[jnp.ndarray, jnp.ndarray]:
    raw = np.loadtxt(Path("tests/fixtures/wavy_points.csv"), delimiter=",", skiprows=1)
    x = jnp.asarray(raw[:, 0:1], dtype=jnp.float32)
    y = jnp.asarray(raw[:, 1], dtype=jnp.float32)
    return x, y


def test_setup_and_devices_returns_runtime_details() -> None:
    report = setup_and_devices()
    assert report.device_count >= 1
    assert len(report.devices) == report.device_count
    assert report.default_backend in {"cpu", "gpu", "tpu"}


def test_numpy_but_jax_demo_matches_expected_shape() -> None:
    out = numpy_but_jax_demo()
    assert out.shape == (3,)
    assert np.isfinite(np.asarray(out)).all()


def test_numpy_where_demo_only_keeps_positive_transforms() -> None:
    out = numpy_indexing_and_where_demo()
    np.testing.assert_allclose(out[:3], jnp.zeros((3,), dtype=jnp.float32))
    assert float(out[-1]) > float(out[-2])


def test_autodiff_grad_demo() -> None:
    assert autodiff_grad_demo(2.0) == 7.0


def test_autodiff_value_and_grad_demo_matches_manual_formula() -> None:
    x = 0.5
    value, grad = autodiff_value_and_grad_demo(x)
    expected_value = np.sin(x) + 0.1 * x**2
    expected_grad = np.cos(x) + 0.2 * x
    assert np.isclose(value, expected_value, atol=1e-6)
    assert np.isclose(grad, expected_grad, atol=1e-6)


def test_higher_order_grad_demo() -> None:
    x = 1.5
    assert np.isclose(higher_order_grad_demo(x), 12.0 * x**2, atol=1e-6)


def test_jit_demo_shape_and_values() -> None:
    x = jnp.array([0.0, 1.0], dtype=jnp.float32)
    out = jit_demo(x)
    np.testing.assert_allclose(out, jnp.tanh(x) + 0.25 * x)


def test_vmap_demo_vectorizes_scalar_work() -> None:
    xs = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    np.testing.assert_allclose(vmap_demo(xs), jnp.array([2.0, 5.0, 10.0]))


def test_vmap_pairwise_l2_demo() -> None:
    points = jnp.array([[0.0, 0.0], [3.0, 4.0]], dtype=jnp.float32)
    reference = jnp.array([0.0, 0.0], dtype=jnp.float32)
    out = vmap_pairwise_l2_demo(points, reference)
    np.testing.assert_allclose(out, jnp.array([0.0, 5.0], dtype=jnp.float32))


def test_pmap_demo_runs_on_available_devices() -> None:
    xs = jnp.arange(max(1, jax.device_count()), dtype=jnp.float32)
    out = pmap_demo(xs)
    np.testing.assert_allclose(out, xs + 1.0)


def test_pmap_center_demo_outputs_zero_mean() -> None:
    xs = jnp.arange(max(1, jax.device_count()), dtype=jnp.float32)
    centered = pmap_center_demo(xs)
    assert np.isclose(float(jnp.mean(centered)), 0.0, atol=1e-6)


def test_prng_demo_is_reproducible_for_same_seed() -> None:
    a1, b1 = prng_demo(seed=7)
    a2, b2 = prng_demo(seed=7)
    np.testing.assert_allclose(a1, a2)
    np.testing.assert_allclose(b1, b2)


def test_prng_dropout_mask_demo_shape_and_dtype() -> None:
    mask = prng_dropout_mask_demo(seed=3, shape=(2, 4), drop_rate=0.25)
    assert mask.shape == (2, 4)
    assert mask.dtype == jnp.bool_


def test_pure_function_step_reduces_mse() -> None:
    x, y = _load_linear_data()
    w0 = jnp.array([0.0], dtype=jnp.float32)

    def mse(w: jnp.ndarray) -> float:
        return float(jnp.mean((x @ w - y) ** 2))

    before = mse(w0)
    w1 = pure_function_step(w0, x, y, lr=0.05)
    assert mse(w1) < before


def test_pytree_demo_scales_nested_arrays() -> None:
    doubled = pytree_demo()
    assert float(doubled["layer1"]["w"][0]) == 2.0
    assert float(doubled["layer2"]["b"][0]) == -1.0


def test_pytree_flatten_demo() -> None:
    leaf_count, leaf_sizes = pytree_flatten_demo()
    assert leaf_count == 3
    assert tuple(sorted(leaf_sizes)) == (1, 2, 4)


def test_training_loop_demo_learns_linear_relationship() -> None:
    x, y = _load_linear_data()
    weights, history = training_loop_demo(x, y, steps=250, lr=0.05)
    assert len(history) == 250
    assert history[-1] < history[0]
    assert 1.7 < float(weights[0]) < 1.95


def test_training_loop_with_validation_demo_runs_on_shifted_data() -> None:
    train_x, train_y = _load_linear_data()
    val_x, val_y = _load_wave_data()
    _, metrics = training_loop_with_validation_demo(
        train_x,
        train_y,
        val_x,
        val_y,
        steps=80,
        lr=0.03,
    )
    assert len(metrics.train_loss) == 80
    assert len(metrics.val_loss) == 80
    assert metrics.train_loss[-1] < metrics.train_loss[0]
