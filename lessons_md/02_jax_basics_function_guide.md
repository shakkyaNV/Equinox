# JAX Basics Function Guide (`jax_learning.basics`)

This is the "what each function does" cheat sheet for the module.

## Setup and devices

- `setup_and_devices()`
  - Collects JAX runtime details (`device_count`, backend, and the visible devices list).
  - Use this first when you are unsure if JAX is seeing CPU/GPU/TPU.

## NumPy-style operations

- `numpy_but_jax_demo()`
  - Uses `jnp.arange`, reshape, broadcasting, and reduction in one compact example.
  - Returns one value per row after centering/scaling.

- `numpy_indexing_and_where_demo()`
  - Demonstrates vectorized masking with `jnp.where`.
  - Applies `log1p` only to positive entries and zeroes out others.

## Autodiff

- `autodiff_grad_demo(x)`
  - Computes the gradient of `x^2 + 3x + 2` at `x`.

- `autodiff_value_and_grad_demo(x)`
  - Computes both objective value and derivative at once using `jax.value_and_grad`.

- `higher_order_grad_demo(x)`
  - Computes a second derivative (`d²/dx² x^4`) to show higher-order autodiff.

## JIT

- `jit_demo(x)`
  - Wraps a small non-linear transform with `@jax.jit`.
  - Same math, compiled execution.

## Vectorization and parallelism

- `vmap_demo(xs)`
  - Converts a scalar function into a batch function with `jax.vmap`.

- `vmap_pairwise_l2_demo(points, reference)`
  - Vectorizes a point-to-reference distance calculation.

- `pmap_demo(xs)`
  - Runs one shard per device using `jax.pmap`, returns `xs + 1` for each shard.

- `pmap_center_demo(xs)`
  - Another `pmap` experiment: centers each shard around its local mean.

## PRNG

- `prng_demo(seed)`
  - Splits one key into two keys and samples from normal/uniform distributions.
  - Demonstrates explicit randomness flow.

- `prng_dropout_mask_demo(seed, shape, drop_rate)`
  - Creates a boolean dropout mask with a chosen keep probability.

## Pure functions and pytrees

- `pure_function_step(weights, x, y, lr)`
  - One gradient descent update for linear regression.
  - No hidden mutation; returns new weights.

- `pytree_demo()`
  - Uses `tree_map` to apply an operation across nested dict parameters.

- `pytree_flatten_demo()`
  - Flattens a pytree and returns leaf count/leaf sizes.

## Training loops

- `training_loop_demo(x, y, steps, lr)`
  - Full tiny training loop for linear regression and its MSE history.

- `training_loop_with_validation_demo(train_x, train_y, val_x, val_y, steps, lr)`
  - Same idea, but tracks both training and validation losses.
