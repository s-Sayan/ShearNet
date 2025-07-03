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

class GalaxyResNet(nn.Module):
    @nn.compact
    def __call__(self, x, deterministic: bool = False):
    
        if x.ndim == 2:
            x = jnp.expand_dims(x, axis=0)
        assert x.ndim == 3, f"Expected input with 3 dimensions (batch_size, height, width), got {x.shape}"

        x = jnp.expand_dims(x, axis=-1)

        # Fewer scales, smaller filters, but with residuals
        x = MultiScaleResidualBlock(filters_per_scale=16, scales=(3, 9, 21))(x)  # 48 total
        x = nn.avg_pool(x, (2, 2), (2, 2))

        x = MultiScaleResidualBlock(filters_per_scale=32, scales=(3, 9, 21))(x)  # 96 total  
        x = nn.avg_pool(x, (2, 2), (2, 2))

        x = x.reshape((x.shape[0], -1))  # 16,224 features (13×13×96)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(4)(x)
        return x