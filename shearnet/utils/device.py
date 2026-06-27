import jax

def get_device():
    """Return (and print) the default JAX device, i.e. the CPU or GPU in use."""
    device = jax.devices()[0]
    print(f"Running on device: {device}")
    return device