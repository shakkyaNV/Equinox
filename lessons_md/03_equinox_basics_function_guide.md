# Equinox Basics Function Guide (`jax_learning.equinox_basics`)

A practical map of what each Equinox function/class in this repo is doing.

## Data containers

- `EquinoxTrainSummary`
  - Dataclass for compact training output (`losses`, learned `weight`, learned `bias`).

## Modules

- `TinyLinear`
  - Minimal Equinox model with one `weight` and one `bias`.
  - `__call__` computes `x * weight + bias`.

- `TinyMLP`
  - Wraps `eqx.nn.MLP` to show nested modules.
  - Uses a PRNG key in constructor for initialization.

- `module_demo(x)`
  - Instantiates `TinyLinear` and runs a forward pass.

- `module_parameter_count_demo()`
  - Counts trainable array elements via tree leaves.

## Filters

- `filter_demo()`
  - Uses `eqx.partition(..., eqx.is_array)` to split dynamic array fields from static metadata.

- `filter_freeze_bias_demo()`
  - Demonstrates a freezing pattern where bias is static (non-array), and weight stays trainable.

## Optax + Equinox training

- `train_with_optax(x, y, steps, learning_rate)`
  - End-to-end training loop:
    - defines loss with `@eqx.filter_value_and_grad`
    - compiles step with `@eqx.filter_jit`
    - applies Optax updates with `eqx.apply_updates`

- `train_with_optax_summary(x, y, steps, learning_rate)`
  - Wrapper over `train_with_optax` returning a summarized dataclass.

## Extra experiment

- `mlp_forward_demo(seed)`
  - Creates a tiny MLP and runs a two-example batch through `jax.vmap`.
