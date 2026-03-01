import jax
import jax.numpy as jnp


def main():
    x = jnp.array([1, 2, 3])
    y = jnp.sin(x)

    print(f"Current JAX device: {jax.devices()}")
    print(f"Y: {y}")


if __name__ == "__main__":
    main()
