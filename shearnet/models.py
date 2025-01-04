import flax.linen as nn
import jax.numpy as jnp

class SimpleGalaxyNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        if x.ndim == 2:  # If batch dimension is missing
            x = jnp.expand_dims(x, axis=0)
        assert x.ndim == 3, f"Expected input with 3 dimensions (batch_size, height, width), got {x.shape}"
        x = jnp.reshape(x, (x.shape[0], -1))  # Flatten
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(2)(x)  # Output e1, e2
        return x

class EnhancedGalaxyNN(nn.Module):
    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        if x.ndim == 2:  # If batch dimension is missing
            x = jnp.expand_dims(x, axis=0)
        assert x.ndim == 3, f"Expected input with 3 dimensions (batch_size, height, width), got {x.shape}"
        
        # Convolutional layers for feature extraction
        x = nn.Conv(32, (3, 3))(x)  # 32 filters, 3x3 kernel
        x = nn.relu(x)
        x = nn.Conv(64, (3, 3))(x)
        x = nn.relu(x)
        print(x.shape)
        x = jnp.reshape(x, (x.shape[0], -1))
        print(x.shape)

        # Fully connected layers        
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        #x = nn.Dropout(0.5, deterministic=deterministic)(x)  # Dropout for regularization
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(2)(x)  # Output e1, e2
        return x