import flax.linen as nn
import jax.numpy as jnp

class ResidualBlock(nn.Module):
    filters: int
    kernel_size: tuple = (3, 3)
    
    @nn.compact
    def __call__(self, x):
        residual = x  # Save the input for the skip connection

        # Ensure residual has the same number of channels as the output
        if x.shape[-1] != self.filters:
            residual = nn.Conv(features=self.filters, kernel_size=(1, 1))(residual)

        # First convolutional layer
        x = nn.Conv(self.filters, self.kernel_size)(x)
        x = nn.leaky_relu(x, negative_slope=0.01)

        # Second convolutional layer
        x = nn.Conv(self.filters, self.kernel_size)(x)

        # Add the residual (skip connection)
        x = x + residual
        x = nn.leaky_relu(x, negative_slope=0.01) # Activation after residual addition
        return x


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
    def __call__(self, x, deterministic: bool = True):
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
        x = nn.Dense(2)(x)
        
        return x
    
    
class GalaxyResNet(nn.Module):
    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        if x.ndim == 2:  # If batch dimension is missing
            x = jnp.expand_dims(x, axis=0)
        assert x.ndim == 3, f"Expected input with 3 dimensions (batch_size, height, width), got {x.shape}"
        x = nn.Conv(32, (3, 3))(x)  # First convolution (32 filters)
        x = nn.leaky_relu(x, negative_slope=0.01)
        print(f"Shape before resnet: {x.shape}")
        # Use ResidualBlocks for feature extraction
        x = ResidualBlock(64)(x)  # First residual block with 64 filters
        x = ResidualBlock(128)(x)  # Second residual block with 128 filters

        '''print(f"Before pooling: {x.shape}")
        
        # Ensure even dimensions with padding
        x = jnp.pad(x, pad_width=((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
        print(f"After padding: {x.shape}")

        # Pooling with window_shape (2, 2) and strides (2, 2)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        print(f"After pooling: {x.shape}")'''
       
        # Flatten the output of the conv layers for the fully connected layers
        x = jnp.reshape(x, (x.shape[0], -1))
        print(f"Shape after resnet: {x.shape}")

        # Ensure that the flattened dimension does not exceed ~200
        #assert x.shape[1] <= 200, f"Flattened dimension is too large: {x.shape[1]}"

        # Fully connected layers        
        x = nn.Dense(128)(x)
        x = nn.leaky_relu(x, negative_slope=0.01)
        # x = nn.Dropout(0.5, deterministic=deterministic)(x)  # Dropout for regularization
        x = nn.Dense(64)(x)
        x = nn.leaky_relu(x, negative_slope=0.01)
        x = nn.Dense(2)(x)  # Output e1, e2
        x = nn.tanh(x)
        return x