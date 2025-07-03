import flax.linen as nn
import jax.numpy as jnp

class MultiScaleResidualBlock(nn.Module):
    filters_per_scale: int
    scales: tuple

    @nn.compact
    def __call__(self, x):
        residual = x

        # Multi-scale convolutions in parallel
        scale_outputs = []
        for scale in self.scales:
            scale_out = nn.Conv(self.filters_per_scale, (scale, scale), padding='SAME')(x)
            scale_outputs.append(scale_out)
        
        # Concatenate multi-scale features
        x = jnp.concatenate(scale_outputs, axis=-1)
        x = nn.relu(x)
        
        # Channel matching for residual
        total_filters = self.filters_per_scale * len(self.scales)
        if residual.shape[-1] != total_filters:
            residual = nn.Conv(total_filters, (1, 1))(residual)

        # Residual connection
        return x + residual

class EnhancedGalaxyNN(nn.Module):
    """
    CNN from Sayan. Changes are listed below:
    - multi-scale feature detection per convolution layer
    - increase in channels per convoluton layer resulting in an increase in channels from 6,272 -> 62,720
    """
    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        # Input handling - same as original
        if x.ndim == 2:
            x = jnp.expand_dims(x, axis=0)
        assert x.ndim == 3, f"Expected input with 3 dimensions (batch_size, height, width), got {x.shape}"

        x = jnp.expand_dims(x, axis=-1)

        # Multi-scale first layer instead of single 3x3
        x_fine = nn.Conv(32, (3, 3), padding='SAME')(x)     # Fine features (original)
        x_small = nn.Conv(32, (5, 5), padding='SAME')(x)    # Small-scale patterns
        x_med = nn.Conv(32, (9, 9), padding='SAME')(x)      # Medium-scale patterns
        x_large = nn.Conv(32, (15, 15), padding='SAME')(x)  # Large-scale patterns
        x_global = nn.Conv(32, (21, 21), padding='SAME')(x) # Global elliptical shape

        # Concatenate multi-scale features
        x = jnp.concatenate([x_fine, x_small, x_med, x_large, x_global], axis=-1)
        print(x.shape)
        x = nn.relu(x)

        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))  # ~27x27x160

        x2_fine = nn.Conv(64, (3, 3), padding='SAME')(x)     # Fine features
        x2_small = nn.Conv(64, (5, 5), padding='SAME')(x)    # Small-scale patterns
        x2_med = nn.Conv(64, (9, 9), padding='SAME')(x)      # Medium-scale patterns
        x2_large = nn.Conv(64, (15, 15), padding='SAME')(x)  # Large-scale patterns
        x2_global = nn.Conv(64, (21, 21), padding='SAME')(x) # Global elliptical shape

        x = jnp.concatenate([x2_fine, x2_small, x2_med, x2_large, x2_global], axis=-1)
        x = nn.relu(x)

        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))  # ~14x14x96

        # Flatten: 14x14x320 = 62,720 features
        x = x.reshape((x.shape[0], -1))

        # Dense layers
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(4)(x)

        return x