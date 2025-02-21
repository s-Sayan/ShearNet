import flax.linen as nn
import jax.numpy as jnp

class ResidualBlock(nn.Module):
    filters: int
    kernel_size: tuple = (3, 3)
    train: bool = True

    """
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
    """
    @nn.compact
    def __call__(self, x):
        residual = x  # Save the input for the skip connection

        # Ensure residual has the same number of channels as the output
        if x.shape[-1] != self.filters:
            residual = nn.Conv(features=self.filters, kernel_size=(1, 1))(residual)

        # First convolutional layer
        x = nn.Conv(self.filters, self.kernel_size)(x)
        x = nn.BatchNorm(use_running_average=not self.train)(x)
        x = nn.leaky_relu(x, negative_slope=0.01)

        # Second convolutional layer
        x = nn.Conv(self.filters, self.kernel_size)(x)
        x = nn.BatchNorm(use_running_average=not self.train)(x)

        # Add the residual (skip connection)
        x = x + residual
        x = nn.leaky_relu(x, negative_slope=0.01)  # Activation after residual addition
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
        x = nn.Dense(2)(x)  
        #x = nn.tanh(x) # Output e1, e2
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

class VGG16(nn.Module):
    @nn.compact
    def __call__(self, x, deterministic: bool=False, train: bool=True):
        if x.ndim == 2:  # If batch dimension is missing
            x = jnp.expand_dims(x, axis=0)
        assert x.ndim == 3, f"Expected input with 3 dimensions (batch_size, height, width), got {x.shape}"
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))

        # VGG16 structure
        x = nn.Conv(64,(3,3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        print(x.shape)
        x = nn.Conv(64,(3,3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        print(x.shape)
        x = nn.max_pool(x,(2,2),(2,2), padding="SAME")
        print(x.shape)
        x = nn.Conv(128,(3,3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        print(x.shape)
        x = nn.Conv(128,(3,3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        print(x.shape)
        x = nn.max_pool(x,(2,2),(2,2), padding="SAME")
        print(x.shape)
        x = nn.Conv(256,(3,3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        print(x.shape)
        x = nn.Conv(256,(3,3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        print(x.shape)
        x = nn.Conv(256,(3,3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        print(x.shape)
        x = nn.max_pool(x,(2,2),(2,2), padding="SAME")
        print(x.shape)
        x = nn.Conv(512,(3,3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        print(x.shape)
        x = nn.Conv(512,(3,3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        print(x.shape)
        x = nn.Conv(512,(3,3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        print(x.shape)
        x = nn.max_pool(x,(2,2),(2,2), padding="SAME")
        print(x.shape)
        x = nn.Conv(512, (3, 3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        print(x.shape)
        x = nn.Conv(512, (3, 3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        print(x.shape)
        x = nn.Conv(512, (3, 3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        print(x.shape)
        x = nn.max_pool(x,(2, 2), (2,2), padding="SAME")
        print(x.shape)

        # flatten
        # print(x.shape)
        x = jnp.reshape(x, (x.shape[0], -1))
        print(f"Shape after vgg16: {x.shape}")

        # Fully connected layers
        x = nn.Dense(256)(x)
        x = nn.leaky_relu(x, negative_slope=0.01)
        x = nn.Dense(128)(x)
        x = nn.leaky_relu(x, negative_slope=0.01)
        x = nn.Dense(2)(x)
        x = nn.tanh(x)

        return x

class ForkCNN(nn.Module):

    @nn.compact
    def __call__(self, x, deterministic: bool=False, train: bool=True):
        if x.ndim == 2:  # If batch dimension is missing
            x = jnp.expand_dims(x, axis=0)
        assert x.ndim == 3, f"Expected input with 3 dimensions (batch_size, height, width), got {x.shape}"
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))

        # ResNet 34
        x = nn.Conv(features=64, kernel_size=(3,), strides=2, padding=3, use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x) # 64
        x = nn.relu(x)

        x = nn.max_pool(x, (3,), (2,))
        x = ResidualBlock(64, train)(x)
        x = ResidualBlock(64, train)(x)
        x = ResidualBlock(64, train)(x)

        x = ResidualBlock(128, train)(x)
        x = ResidualBlock(128, train)(x)
        x = ResidualBlock(128, train)(x)
        x = ResidualBlock(128, train)(x)

        x = ResidualBlock(256, train)(x)
        x = ResidualBlock(256, train)(x)
        x = ResidualBlock(256, train)(x)
        x = ResidualBlock(256, train)(x)
        x = ResidualBlock(256, train)(x)
        x = ResidualBlock(256, train)(x)

        x = ResidualBlock(512, train)(x)
        x = ResidualBlock(512, train)(x)
        x = ResidualBlock(512, train)(x)

        x = nn.avg_pool(x, (2,2))

        # CNN layer
        x = nn.Conv(features=32, kernel_size=(3,), strides=2, padding=3, use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x) # 32
        x = nn.relu(x)

        x = nn.Conv(features=64, kernel_size=(3,), strides=2, padding=3, use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x) # 64
        x = nn.relu(x)

        x = nn.Conv(features=32, kernel_size=(3,), strides=2, padding=3, use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x) #32
        x = nn.relu(x)

        x = nn.Conv(features=16, kernel_size=(3,), strides=2, padding=3, use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x) #16
        x = nn.relu(x)

        x = jnp.reshape(x, (x.shape[0], -1))
        print(f"Shape after ForkCNN: {x.shape}")

        # Fully connected layer
        x = nn.Dense(512)(x)
        x = nn.Dense(128)(x)
        x = nn.Dense(2)(x)
        x = nn.tanh(x)

        return x
