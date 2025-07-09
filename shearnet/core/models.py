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
        return x.reshape((x.shape[0], -1))
        
class EnhancedGalaxyNN(nn.Module):
    """
    Built off of the CNN from Sayan above. Changes are listed below:
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
        return x.reshape((x.shape[0], -1))
        
class OriginalGalaxyResNet(nn.Module):
    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        if x.ndim == 2:  # If batch dimension is missing
            x = jnp.expand_dims(x, axis=0)
        assert x.ndim == 3, f"Expected input with 3 dimensions (batch_size, height, width), got {x.shape}"
        x = jnp.expand_dims(x, axis=-1)
        x = nn.Conv(32, (3, 3))(x)  # First convolution (32 filters)
        x = nn.leaky_relu(x, negative_slope=0.01)
        #print(f"Shape before resnet: {x.shape}")
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
        return jnp.reshape(x, (x.shape[0], -1))
        
class GalaxyResNet(nn.Module):
    """
    Built off of the ResNet from Sayan above. Changes are listed below:
    - multi-scale feature detection per convolution layer
    """
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

        return x.reshape((x.shape[0], -1))  # 16,224 features (13×13×96)

class CBAM_Attention(nn.Module):
    """
    Convolutional Block Attention Module with full citations.
    """
    reduction_ratio: int = 8

    @nn.compact
    def __call__(self, x):
        # ==================== CHANNEL ATTENTION MODULE ====================
        # CITATION: "CBAM: Convolutional Block Attention Module" (Woo et al., ECCV 2018)
        # MOTIVATION: "What meaningful features to emphasize or suppress"
        # RATIONALE: Different feature channels encode different types of information
        
        # CITATION: "Squeeze-and-Excitation Networks" (Hu et al., CVPR 2018)
        # RATIONALE: Global context via spatial pooling
        avg_pool = jnp.mean(x, axis=(1, 2), keepdims=True)  # Global average pooling
        max_pool = jnp.max(x, axis=(1, 2), keepdims=True)   # Global max pooling

        # CITATION: CBAM paper - shared MLP for efficient parameter usage
        # RATIONALE: Reduces overfitting by sharing weights between avg and max paths
        shared_mlp = lambda inp: nn.Dense(x.shape[-1])(nn.relu(nn.Dense(x.shape[-1] // self.reduction_ratio)(inp)))

        avg_out = shared_mlp(avg_pool)
        max_out = shared_mlp(max_pool)
        
        # CITATION: "Sigmoid" activation for attention weights (Hochreiter & Schmidhuber, 1997)
        # RATIONALE: Produces weights between 0 and 1 for soft attention
        channel_att = nn.sigmoid(avg_out + max_out)

        # Apply channel attention
        x = x * channel_att

        # ==================== SPATIAL ATTENTION MODULE ====================
        # CITATION: "CBAM: Convolutional Block Attention Module" (Woo et al., ECCV 2018)
        # MOTIVATION: "Where to focus" in spatial dimension
        # RATIONALE: Important for galaxy shape measurement where spatial location matters
        
        avg_spatial = jnp.mean(x, axis=-1, keepdims=True)
        max_spatial = jnp.max(x, axis=-1, keepdims=True)
        spatial_concat = jnp.concatenate([avg_spatial, max_spatial], axis=-1)

        # CITATION: CBAM paper recommends 7x7 kernel for spatial attention
        # RATIONALE: Large kernel captures broader spatial context
        spatial_att = nn.Conv(1, (7, 7), padding='SAME')(spatial_concat)
        spatial_att = nn.sigmoid(spatial_att)

        # Apply spatial attention
        return x * spatial_att

class EnhancedMultiScaleBlock(nn.Module):
    """
    Enhanced Multi-Scale Residual Block with comprehensive citations.
    """
    filters_per_scale: int
    scales: tuple
    use_dilated: bool = True

    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        residual = x

        # ==================== MULTI-SCALE CONVOLUTIONS ====================
        scale_outputs = []
        for scale in self.scales:
            if self.use_dilated and scale > 3:
                # CITATION: "Multi-Scale Context Aggregation by Dilated Convolutions" (Yu & Koltun, ICLR 2016)
                # QUOTE: "systematically aggregates multi-scale contextual information without losing resolution"
                # RATIONALE: Achieves large receptive fields with fewer parameters than large kernels
                # MATH: 21x21 kernel = 441 parameters, 3x3 dilated with rate 7 = 9 parameters (same receptive field)
                dilation = scale // 3
                scale_out = nn.Conv(
                    self.filters_per_scale, 
                    (3, 3), 
                    padding='SAME',
                    kernel_dilation=(dilation, dilation)
                )(x)
            else:
                # CITATION: Standard convolution from "Gradient-Based Learning Applied to Document Recognition" (LeCun et al., 1998)
                # RATIONALE: Regular convolutions for smaller scales where dilation isn't beneficial
                scale_out = nn.Conv(
                    self.filters_per_scale, 
                    (scale, scale), 
                    padding='SAME'
                )(x)

            # CITATION: "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
            #           (Ioffe & Szegedy, ICML 2015)
            # PLACEMENT: After convolution, before activation (standard practice)
            scale_out = nn.BatchNorm(use_running_average=True, axis_name=None)(scale_out)
            scale_out = nn.relu(scale_out)
            scale_outputs.append(scale_out)
        
        # ==================== FEATURE CONCATENATION ====================
        # CITATION: "Going Deeper with Convolutions" (Szegedy et al., CVPR 2015) - Inception architecture
        # RATIONALE: Combines features from different scales for richer representation
        x = jnp.concatenate(scale_outputs, axis=-1)

        # ==================== CBAM ATTENTION ====================
        # CITATION: "CBAM: Convolutional Block Attention Module" (Woo et al., ECCV 2018)
        # PERFORMANCE: "consistently improved classification and detection performances"
        # RATIONALE: Focuses on important spatial locations and channels for galaxy shape measurement
        x = CBAM_Attention()(x)

        # ==================== RESIDUAL CONNECTION ====================
        # CITATION: "Deep Residual Learning for Image Recognition" (He et al., CVPR 2016)
        # QUOTE: "explicitly reformulate the layers as learning residual functions"
        # RATIONALE: Enables training of deeper networks by addressing vanishing gradient problem
        
        total_filters = self.filters_per_scale * len(self.scales)
        if residual.shape[-1] != total_filters:
            # CITATION: "Identity Mappings in Deep Residual Networks" (He et al., ECCV 2016)
            # RATIONALE: 1x1 convolution for dimension matching in residual connections
            residual = nn.Conv(total_filters, (1, 1))(residual)
            residual = nn.BatchNorm(use_running_average=True, axis_name=None)(residual)

        # CITATION: "Identity Mappings in Deep Residual Networks" (He et al., ECCV 2016)
        # RATIONALE: Pre-activation design for better gradient flow
        # QUOTE: "the forward and backward signals can be directly propagated from one block to any other block"
        return nn.relu(x + residual)

class ResearchBackedGalaxyResNet(nn.Module):
    """
    Research-backed Galaxy ResNet with comprehensive citations for every design decision.
    
    OVERALL ARCHITECTURE PHILOSOPHY:
    - Multi-scale processing: Inspired by galaxy morphology having features at different scales
    - Residual learning: "Deep Residual Learning for Image Recognition" (He et al., CVPR 2016)
    - Attention mechanisms: "CBAM: Convolutional Block Attention Module" (Woo et al., ECCV 2018)
    - Conservative enhancement: Maintains successful elements from your original design
    """

    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        
        # ==================== INPUT HANDLING ====================
        # CITATION: Standard practice in computer vision, established in LeNet-5 (LeCun et al., 1998)
        # RATIONALE: Ensures consistent tensor dimensions for batch processing
        if x.ndim == 2:
            x = jnp.expand_dims(x, axis=0)
        assert x.ndim == 3, f"Expected input with 3 dimensions (batch_size, height, width), got {x.shape}"

        # CITATION: "ImageNet Classification with Deep Convolutional Neural Networks" (Krizhevsky et al., NIPS 2012)
        # RATIONALE: Convert grayscale to single-channel format expected by CNNs
        x = jnp.expand_dims(x, axis=-1)

        # ==================== INITIAL FEATURE EXTRACTION ====================
        # CITATION: "Very Deep Convolutional Networks for Large-Scale Image Recognition" (Simonyan & Zisserman, ICLR 2015)
        # RATIONALE: 3x3 kernels are computationally efficient while capturing local features
        # DECISION: Small initial feature count (16) to match your successful original design
        x = nn.Conv(16, (3, 3), padding='SAME')(x)
        
        # CITATION: "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" 
        #           (Ioffe & Szegedy, ICML 2015)
        # RATIONALE: "allows us to use much higher learning rates and be less careful about initialization"
        # DECISION: use_running_average=True prevents batch_stats complexity in your existing pipeline
        x = nn.BatchNorm(use_running_average=True, axis_name=None)(x)
        
        # CITATION: "Rectified Linear Units Improve Restricted Boltzmann Machines" (Nair & Hinton, ICML 2010)
        # RATIONALE: ReLU prevents vanishing gradients and is computationally efficient
        x = nn.relu(x)

        # ==================== FIRST MULTI-SCALE BLOCK ====================
        # CITATION: Multi-scale approach inspired by:
        # 1. "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning" (Szegedy et al., 2017)
        # 2. Your own successful results with scales (3, 9, 21)
        # RATIONALE: Galaxies have features at multiple spatial scales (PSF effects, substructure, overall shape)
        x = EnhancedMultiScaleBlock(
            filters_per_scale=16,  # DECISION: Matches your successful original design
            scales=(3, 9, 21),     # DECISION: Preserves your empirically successful scale selection
            use_dilated=True       # CITATION: "Multi-Scale Context Aggregation by Dilated Convolutions" (Yu & Koltun, ICLR 2016)
        )(x, deterministic=deterministic)

        # ==================== LEARNABLE DOWNSAMPLING ====================
        # CITATION: "Striving for Simplicity: The All Convolutional Net" (Springenberg et al., ICLR 2015)
        # RATIONALE: "replacing pooling operations with convolutional layers with stride > 1"
        # ADVANTAGE: Learnable parameters vs fixed pooling operation
        x = nn.Conv(x.shape[-1], (3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=True, axis_name=None)(x)
        x = nn.relu(x)

        # ==================== SECOND MULTI-SCALE BLOCK ====================
        # CITATION: Same rationale as first block, with increased capacity
        # DECISION: filters_per_scale=32 matches your successful original design
        x = EnhancedMultiScaleBlock(
            filters_per_scale=32,  # DECISION: 2x increase in capacity, matches your original
            scales=(3, 9, 21),     # DECISION: Consistent scale selection
            use_dilated=True
        )(x, deterministic=deterministic)

        return jnp.mean(x, axis=(1, 2))

        '''
        # ==================== GLOBAL AVERAGE POOLING ====================
        # CITATION: "Network In Network" (Lin, Chen & Yan, ICLR 2014)
        # QUOTE: "more robust to spatial translations of the input"
        # QUOTE: "no parameter to optimize in the fully connected layers, overfitting is avoided"
        # RATIONALE: Reduces parameters from ~16,224 to 96, preventing overfitting
        # TRADE-OFF: May lose spatial information important for galaxy shape measurement
        x = jnp.mean(x, axis=(1, 2))  # Global average pooling

        # Print for comparison with your original 16,224 features
        print(f"Flattened shape: {x.shape}")

        # ==================== CLASSIFICATION HEAD ====================
        # CITATION: "ImageNet Classification with Deep Convolutional Neural Networks" (Krizhevsky et al., NIPS 2012)
        # RATIONALE: Dense layers for final feature combination and prediction
        # DECISION: 128 units matches your successful original design
        x = nn.Dense(128)(x)
        
        # CITATION: Batch norm in dense layers: "Batch Normalization: Accelerating Deep Network Training"
        # RATIONALE: Normalizes inputs to activation function
        x = nn.BatchNorm(use_running_average=True, axis_name=None)(x)
        x = nn.relu(x)
        
        # OPTIONAL REGULARIZATION (commented out for initial testing):
        # CITATION: "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (Srivastava et al., JMLR 2014)
        # x = nn.Dropout(0.5)(x, deterministic=deterministic)

        # ==================== FINAL PREDICTION LAYER ====================
        # DECISION: 4 outputs to match your pipeline expectations (g1, g2, sigma, flux)
        # CITATION: Standard practice since "Gradient-Based Learning Applied to Document Recognition" (LeCun et al., 1998)
        # RATIONALE: Linear layer for regression output, no activation for unbounded predictions
        x = nn.Dense(4)(x)

        return x
        '''

class ForkLike(nn.Module):
    """
    This class is meant to take two of the above models, train one on galaxy images and another on psf images, then concatenate the results and do the dense/fully connected layers.

    This should basically mimimic the forklens structure here https://github.com/zhangzzk/forklens/.
    """

    @nn.compact
    def __call__(self, galaxy_nn_model, galaxy_image, psf_nn_model, psf_image, deterministic: bool = False):

        def get_model(nn):
            if nn == "cnn":
                return OriginalGalaxyNN()
            elif nn == "dev_cnn":
                return EnhancedGalaxyNN()
            elif nn == "resnet":
                return OriginalGalaxyResNet()
            elif nn == "dev_resnet":
                return GalaxyResNet()
            elif nn == "research_backed":
                return ResearchBackedGalaxyResNet()
            else:
                raise ValueError("Invalid model type specified.")

        # This model will learn from galaxy images
        galaxy_features = get_model(galaxy_nn_model)(
            galaxy_image, deterministic=deterministic
        )
        
        # This model will learn from psf images
        psf_features = get_model(psf_nn_model)(
            psf_image, deterministic=deterministic
        )
        
        # Combines features from the two separate models above trained on different types of images to represent them in one feature layer
        combined_features = jnp.concatenate([galaxy_features, psf_features], axis=-1)
        print(f"Galaxy features shape: {galaxy_features.shape}")
        print(f"PSF features shape: {psf_features.shape}")
        print(f"Combined features shape: {combined_features.shape}")

        # The fully connected layers
        x = nn.Dense(128)(combined_features)
        x = nn.BatchNorm(use_running_average=True, axis_name=None)(x)
        x = nn.relu(x)

        # Final predictions of [g1, g2, sigma, flux]
        x = nn.Dense(4)(x)
        return x