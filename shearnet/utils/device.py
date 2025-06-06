import jax

def get_device():
    device = jax.devices()[0]
    print(f"Running on device: {device}")
    return device