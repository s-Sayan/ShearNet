import flax.linen as nn
import jax.numpy as jnp

class OriginalGalaxyNN(nn.Module):
    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        # Input handling 
        if x.ndim == 2:
            x = jnp.expand_dims(x, axis=0)
        assert x.ndim == 3, f"Expected input with 3 dimensions (batch_size, height, width), got {x.shape}"
        
        x = jnp.expand_dims(x, axis=-1)
        
        # Simple conv stack with pooling
        x = nn.Conv(16, (3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))  # 27x27x16
        
        x = nn.Conv(32, (3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))  # 14x14x32
        
        # Flatten: 14*14*32 = 6,272 features
        x = x.reshape((x.shape[0], -1))
        
        # Dense layers similar to working FNN
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        #x = nn.Dropout(0.5)(x, deterministic=deterministic)  # Dropout applied only if deterministic=False
        x = nn.Dense(4)(x)
        #x = 0.5*nn.tanh(x)
        return x
        