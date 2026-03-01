import jax.numpy as jnp


def main() -> None:
    x = jnp.arange(10, dtype=jnp.float32)
    centered = x - x.mean()
    print(f"x: {x}")
    print(f"mean(x): {x.mean():.3f}")
    print(f"centered x: {centered}")


if __name__ == "__main__":
    main()
