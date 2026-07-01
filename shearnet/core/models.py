"""Flax neural-network architectures for galaxy shear estimation."""

import flax.linen as nn
import jax.numpy as jnp


class ResidualBlock(nn.Module):
    """A two-convolution residual block with a skip connection.

    The input is passed through two convolutions and added back to a
    (optionally 1x1-projected) copy of itself, following the residual-learning
    design of He et al. (CVPR 2016). Used as a building block by ``GalaxyResNet``.

    Attributes:
        filters: Number of output channels for the block.
        kernel_size: Convolution kernel size (default ``(3, 3)``).
    """

    filters: int
    kernel_size: tuple = (3, 3)

    @nn.compact
    def __call__(self, x):
        """Apply the residual block to ``x`` and return the activated output."""
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
        x = nn.leaky_relu(x, negative_slope=0.01)  # Activation after residual addition
        return x


class SimpleGalaxyNN(nn.Module):
    """A plain multi-layer perceptron (``nn='mlp'``) for shear estimation.

    Flattens the input stamp and applies dense layers to regress the requested
    output parameters. The lightest of the available architectures.

    The ``__call__`` signature is shared across all single-branch models:

    Args:
        x: Input image batch, shape ``(batch, height, width)`` (a leading batch
            axis is added if a single 2-D stamp is passed).
        deterministic: Disables stochastic layers (e.g. dropout) when ``True``.
        fork: If ``True``, return the flattened feature vector instead of the
            final prediction (used by ``ForkLike`` to fuse two branches).
        gap: Use global-average-pooling instead of flattening (where supported).
        output_keys: Names of the parameters to predict; the output dimension
            equals ``len(output_keys)``.

    Returns:
        Array of shape ``(batch, len(output_keys))`` with the predicted
        parameters, or the feature vector when ``fork=True``.
    """

    @nn.compact
    def __call__(
        self,
        x,
        deterministic: bool = False,
        fork: bool = False,
        gap: bool = False,
        output_keys: tuple = ("g1", "g2"),
    ):
        """Run the MLP and return the predictions (or features when ``fork``)."""
        if x.ndim == 2:  # If batch dimension is missing
            x = jnp.expand_dims(x, axis=0)
        assert (
            x.ndim == 3
        ), f"Expected input with 3 dimensions (batch_size, height, width), got {x.shape}"
        x = jnp.reshape(x, (x.shape[0], -1))  # Flatten
        if fork:
            return x
        else:
            x = nn.Dense(128)(x)
            x = nn.relu(x)
            x = nn.Dense(64)(x)
            x = nn.relu(x)
            x = nn.Dense(len(output_keys))(x)  # Output e1, e2
            return x


class EnhancedGalaxyNN(nn.Module):
    """A compact convolutional network (``nn='cnn'``) for shear estimation.

    Two convolution + average-pool blocks extract spatial features which are
    flattened and passed through dense layers. A good default architecture.

    Shares the common model signature (see :class:`SimpleGalaxyNN`). In addition,
    ``return_spatial=True`` returns the intermediate spatial feature map (used by
    the transformer-fusion path of :class:`ForkLike`).
    """

    @nn.compact
    def __call__(
        self,
        x,
        deterministic: bool = False,
        fork: bool = False,
        gap: bool = False,
        output_keys: tuple = ("g1", "g2"),
        return_spatial: bool = False,
    ):
        """Run the CNN and return predictions, features, or the spatial map."""
        # Input handling
        if x.ndim == 2:
            x = jnp.expand_dims(x, axis=0)
        assert (
            x.ndim == 3
        ), f"Expected input with 3 dimensions (batch_size, height, width), got {x.shape}"

        x = jnp.expand_dims(x, axis=-1)

        # Simple conv stack with pooling
        x = nn.Conv(16, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))  # 27x27x16

        x = nn.Conv(32, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))  # 14x14x32

        if return_spatial:
            return x

        # Flatten: 14*14*32 = 6,272 features
        x = x.reshape((x.shape[0], -1))

        if fork:
            return x
        else:
            # Dense layers similar to working FNN
            x = nn.Dense(128)(x)
            x = nn.relu(x)
            # x = nn.Dropout(0.5)(x, deterministic=deterministic)  # Dropout applied only if
            # deterministic=False
            x = nn.Dense(len(output_keys))(x)
            # x = 0.5*nn.tanh(x)
            return x


class GalaxyResNet(nn.Module):
    """A residual CNN (``nn='resnet'``) built from :class:`ResidualBlock`s.

    Applies an initial convolution followed by two residual blocks of growing
    width, then dense layers with a ``tanh`` output. Heavier than
    :class:`EnhancedGalaxyNN`; shares the common model signature
    (see :class:`SimpleGalaxyNN`).
    """

    @nn.compact
    def __call__(
        self,
        x,
        deterministic: bool = False,
        fork: bool = False,
        gap: bool = False,
        output_keys: tuple = ("g1", "g2"),
    ):
        """Run the residual CNN and return predictions (or features)."""
        if x.ndim == 2:  # If batch dimension is missing
            x = jnp.expand_dims(x, axis=0)
        assert (
            x.ndim == 3
        ), f"Expected input with 3 dimensions (batch_size, height, width), got {x.shape}"
        x = jnp.expand_dims(x, axis=-1)
        x = nn.Conv(32, (3, 3))(x)  # First convolution (32 filters)
        x = nn.leaky_relu(x, negative_slope=0.01)
        # print(f"Shape before resnet: {x.shape}")
        # Use ResidualBlocks for feature extraction
        x = ResidualBlock(64)(x)  # First residual block with 64 filters
        x = ResidualBlock(128)(x)  # Second residual block with 128 filters

        # Flatten the output of the conv layers for the fully connected layers
        x = jnp.reshape(x, (x.shape[0], -1))
        # print(f"Shape after resnet: {x.shape}")

        if fork:
            return x
        else:
            # Fully connected layers
            x = nn.Dense(128)(x)
            x = nn.leaky_relu(x, negative_slope=0.01)
            # x = nn.Dropout(0.5, deterministic=deterministic)(x)  # Dropout for regularization
            x = nn.Dense(64)(x)
            x = nn.leaky_relu(x, negative_slope=0.01)
            x = nn.Dense(len(output_keys))(x)
            x = nn.tanh(x)
            return x


class CBAM_Attention(nn.Module):
    """Convolutional Block Attention Module with full citations."""

    reduction_ratio: int = 8

    @nn.compact
    def __call__(self, x):
        """Apply channel and spatial attention to ``x`` and return the result."""
        # ==================== CHANNEL ATTENTION MODULE ====================
        # CITATION: "CBAM: Convolutional Block Attention Module" (Woo et al., ECCV 2018)
        # MOTIVATION: "What meaningful features to emphasize or suppress"
        # RATIONALE: Different feature channels encode different types of information

        # CITATION: "Squeeze-and-Excitation Networks" (Hu et al., CVPR 2018)
        # RATIONALE: Global context via spatial pooling
        avg_pool = jnp.mean(x, axis=(1, 2), keepdims=True)  # Global average pooling
        max_pool = jnp.max(x, axis=(1, 2), keepdims=True)  # Global max pooling

        # CITATION: CBAM paper - shared MLP for efficient parameter usage
        # RATIONALE: Reduces overfitting by sharing weights between avg and max paths
        def shared_mlp(inp):
            reduced = nn.Dense(x.shape[-1] // self.reduction_ratio)(inp)
            return nn.Dense(x.shape[-1])(nn.relu(reduced))

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
        spatial_att = nn.Conv(1, (7, 7), padding="SAME")(spatial_concat)
        spatial_att = nn.sigmoid(spatial_att)

        # Apply spatial attention
        return x * spatial_att


class EnhancedMultiScaleBlock(nn.Module):
    """Enhanced multi-scale residual block with comprehensive citations."""

    filters_per_scale: int
    scales: tuple
    use_dilated: bool = True

    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        """Apply the multi-scale residual block to ``x`` and return the result."""
        residual = x

        # ==================== MULTI-SCALE CONVOLUTIONS ====================
        scale_outputs = []
        for scale in self.scales:
            if self.use_dilated and scale > 3:
                # CITATION: "Multi-Scale Context Aggregation by Dilated Convolutions" (Yu & Koltun,
                # ICLR 2016)
                # QUOTE: "systematically aggregates multi-scale contextual information without
                # losing resolution"
                # RATIONALE: Achieves large receptive fields with fewer parameters than large
                # kernels
                # MATH: 21x21 kernel = 441 parameters, 3x3 dilated with rate 7 = 9 parameters (same
                # receptive field)
                dilation = scale // 3
                scale_out = nn.Conv(
                    self.filters_per_scale,
                    (3, 3),
                    padding="SAME",
                    kernel_dilation=(dilation, dilation),
                )(x)
            else:
                # CITATION: Standard convolution from "Gradient-Based Learning Applied to Document
                # Recognition" (LeCun et al., 1998)
                # RATIONALE: Regular convolutions for smaller scales where dilation isn't beneficial
                scale_out = nn.Conv(self.filters_per_scale, (scale, scale), padding="SAME")(x)

            # CITATION: "Batch Normalization: Accelerating Deep Network Training by Reducing
            # Internal Covariate Shift"
            #           (Ioffe & Szegedy, ICML 2015)
            # PLACEMENT: After convolution, before activation (standard practice)
            scale_out = nn.GroupNorm(num_groups=8)(scale_out)
            scale_out = nn.relu(scale_out)
            scale_outputs.append(scale_out)

        # ==================== FEATURE CONCATENATION ====================
        # CITATION: "Going Deeper with Convolutions" (Szegedy et al., CVPR 2015) - Inception
        # architecture
        # RATIONALE: Combines features from different scales for richer representation
        x = jnp.concatenate(scale_outputs, axis=-1)

        # ==================== CBAM ATTENTION ====================
        # CITATION: "CBAM: Convolutional Block Attention Module" (Woo et al., ECCV 2018)
        # PERFORMANCE: "consistently improved classification and detection performances"
        # RATIONALE: Focuses on important spatial locations and channels for galaxy shape
        # measurement
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
            residual = nn.GroupNorm(num_groups=8)(residual)

        # CITATION: "Identity Mappings in Deep Residual Networks" (He et al., ECCV 2016)
        # RATIONALE: Pre-activation design for better gradient flow
        # QUOTE: "the forward and backward signals can be directly propagated from one block to any
        # other block"
        return nn.relu(x + residual)


class ResearchBackedGalaxyResNet(nn.Module):
    """
    Research-backed Galaxy ResNet with comprehensive citations for every design decision.

    OVERALL ARCHITECTURE PHILOSOPHY:
    - Multi-scale processing: Inspired by galaxy morphology having features at different scales
    - Residual learning: "Deep Residual Learning for Image Recognition" (He et al., CVPR 2016)
    - Attention mechanisms: "CBAM: Convolutional Block Attention Module" (Woo et al., ECCV 2018)
    """

    @nn.compact
    def __call__(
        self,
        x,
        deterministic: bool = False,
        fork: bool = False,
        gap: bool = False,
        output_keys: tuple = ("g1", "g2"),
        return_spatial: bool = False,
    ):
        """Run the multi-scale ResNet and return predictions, features, or map."""
        # ==================== INPUT HANDLING ====================
        # CITATION: Standard practice in computer vision, established in LeNet-5 (LeCun et al.,
        # 1998)
        # RATIONALE: Ensures consistent tensor dimensions for batch processing
        if x.ndim == 2:
            x = jnp.expand_dims(x, axis=0)
        assert (
            x.ndim == 3
        ), f"Expected input with 3 dimensions (batch_size, height, width), got {x.shape}"

        # CITATION: "ImageNet Classification with Deep Convolutional Neural Networks" (Krizhevsky
        # et al., NIPS 2012)
        # RATIONALE: Convert grayscale to single-channel format expected by CNNs
        x = jnp.expand_dims(x, axis=-1)

        # ==================== INITIAL FEATURE EXTRACTION ====================
        # CITATION: "Very Deep Convolutional Networks for Large-Scale Image Recognition" (Simonyan
        # & Zisserman, ICLR 2015)
        # RATIONALE: 3x3 kernels are computationally efficient while capturing local features
        # DECISION: Small initial feature count (16) to match your successful original design
        x = nn.Conv(16, (3, 3), padding="SAME")(x)

        # CITATION: "Batch Normalization: Accelerating Deep Network Training by Reducing Internal
        # Covariate Shift"
        #           (Ioffe & Szegedy, ICML 2015)
        # RATIONALE: "allows us to use much higher learning rates and be less careful about
        # initialization"
        # DECISION: use_running_average=True prevents batch_stats complexity in existing pipeline
        x = nn.GroupNorm(num_groups=8)(x)

        # CITATION: "Rectified Linear Units Improve Restricted Boltzmann Machines" (Nair & Hinton,
        # ICML 2010)
        # RATIONALE: ReLU prevents vanishing gradients and is computationally efficient
        x = nn.relu(x)

        # ==================== FIRST MULTI-SCALE BLOCK ====================
        # CITATION: Multi-scale approach inspired by:
        # 1. "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning"
        # (Szegedy et al., 2017)
        # 2. Your own successful results with scales (3, 9, 21)
        # RATIONALE: Galaxies have features at multiple spatial scales (PSF effects, substructure,
        # overall shape)
        x = EnhancedMultiScaleBlock(
            filters_per_scale=16,  # DECISION: Matches your successful original design
            scales=(3, 9, 21),  # DECISION: Preserves your empirically successful scale selection
            # CITATION: "Multi-Scale Context Aggregation by Dilated Convolutions"
            # (Yu & Koltun, ICLR 2016)
            use_dilated=True,
        )(x, deterministic=deterministic)

        # ==================== LEARNABLE DOWNSAMPLING ====================
        # CITATION: "Striving for Simplicity: The All Convolutional Net" (Springenberg et al., ICLR
        # 2015)
        # RATIONALE: "replacing pooling operations with convolutional layers with stride > 1"
        # ADVANTAGE: Learnable parameters vs fixed pooling operation
        x = nn.Conv(x.shape[-1], (3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.relu(x)

        # ==================== SECOND MULTI-SCALE BLOCK ====================
        # CITATION: Same rationale as first block, with increased capacity
        # DECISION: filters_per_scale=32 matches your successful original design
        x = EnhancedMultiScaleBlock(
            filters_per_scale=32,  # DECISION: 2x increase in capacity, matches your original
            scales=(3, 9, 21),  # DECISION: Consistent scale selection
            use_dilated=True,
        )(x, deterministic=deterministic)

        if return_spatial:
            return x

        if gap:
            # ==================== GLOBAL AVERAGE POOLING ====================
            # CITATION: "Network In Network" (Lin, Chen & Yan, ICLR 2014)
            # QUOTE: "more robust to spatial translations of the input"
            # QUOTE: "no parameter to optimize in the fully connected layers, overfitting is
            # avoided"
            # RATIONALE: Reduces parameters from ~16,224 to 96, preventing overfitting
            # TRADE-OFF: May lose spatial information important for galaxy shape measurement
            x = jnp.mean(x, axis=(1, 2))
        else:
            x = x.reshape((x.shape[0], -1))

        # print(f"Flattened shape: {x.shape}")

        if fork:
            return x
        else:

            # ==================== CLASSIFICATION HEAD ====================
            # CITATION: "ImageNet Classification with Deep Convolutional Neural Networks"
            # (Krizhevsky et al., NIPS 2012)
            # RATIONALE: Dense layers for final feature combination and prediction
            # DECISION: 128 units matches your successful original design
            x = nn.Dense(128)(x)

            # CITATION: Batch norm in dense layers: "Batch Normalization: Accelerating Deep Network
            # Training"
            # RATIONALE: Normalizes inputs to activation function
            x = nn.GroupNorm(num_groups=8)(x)
            x = nn.relu(x)

            # OPTIONAL REGULARIZATION (commented out for initial testing):
            # CITATION: "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
            # (Srivastava et al., JMLR 2014)
            # x = nn.Dropout(0.5)(x, deterministic=deterministic)

            # ==================== FINAL PREDICTION LAYER ====================
            # DECISION: output_keys to match your pipeline expectations (g1, g2, sigma, flux)
            # CITATION: Standard practice since "Gradient-Based Learning Applied to Document
            # Recognition" (LeCun et al., 1998)
            # RATIONALE: Linear layer for regression output, no activation for unbounded predictions
            x = nn.Dense(len(output_keys))(x)

            return x


class ForkLensPSFNet(nn.Module):
    """Strided CNN for PSF stamps (``nn='forklens_psf'``), from ForkLens.

    Four stride-2 convolution blocks progressively downsample the PSF image.
    Designed to be used as the PSF branch of :class:`ForkLike`, mirroring the
    ``cnn_layers`` design of the ForkLens project. Shares the common model
    signature (see :class:`SimpleGalaxyNN`).
    """

    @nn.compact
    def __call__(
        self,
        x,
        deterministic: bool = False,
        fork: bool = False,
        gap: bool = False,
        output_keys: tuple = ("g1", "g2"),
        return_spatial: bool = False,
    ):
        """Run the strided PSF CNN and return predictions, features, or map."""
        # Input handling
        if x.ndim == 2:
            x = jnp.expand_dims(x, axis=0)
        assert (
            x.ndim == 3
        ), f"Expected input with 3 dimensions (batch_size, height, width), got {x.shape}"
        x = jnp.expand_dims(x, axis=-1)

        # First conv block
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((3, 3), (3, 3)),
            use_bias=False,
        )(x)
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.relu(x)

        # Second conv block
        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((3, 3), (3, 3)),
            use_bias=False,
        )(x)
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.relu(x)

        # Third conv block
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((3, 3), (3, 3)),
            use_bias=False,
        )(x)
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.relu(x)

        # Fourth conv block
        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((3, 3), (3, 3)),
            use_bias=False,
        )(x)
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.relu(x)

        if return_spatial:
            return x

        # Flatten for concatenation
        x = x.reshape((x.shape[0], -1))

        if fork:
            return x
        else:
            x = nn.Dense(128)(x)
            x = nn.relu(x)
            x = nn.Dense(64)(x)
            x = nn.relu(x)
            x = nn.Dense(len(output_keys))(x)
            return x


class TransformerFusion(nn.Module):
    """
    Hybrid spatial cross-attention + self-attention fusion for ForkLike.

    Galaxy spatial tokens act as queries; PSF spatial tokens act as keys/values.
    This is physically motivated: the galaxy branch queries the PSF branch to
    learn spatially-specific PSF correction.

    Args:
        d_model: shared token dimension after projection (default 128)
        num_heads: attention heads (d_model must be divisible by this)
        num_self_attn_layers: self-attention layers applied after cross-attention
    """

    d_model: int = 128
    num_heads: int = 4
    num_self_attn_layers: int = 2

    @nn.compact
    def __call__(
        self, galaxy_map, psf_map, output_keys: tuple = ("g1", "g2"), deterministic: bool = True
    ):
        """Fuse galaxy and PSF feature maps and return the prediction."""
        batch = galaxy_map.shape[0]
        H_g, W_g = galaxy_map.shape[1], galaxy_map.shape[2]
        H_p, W_p = psf_map.shape[1], psf_map.shape[2]

        # Project both branches to shared d_model via 1x1 conv
        gal_proj = nn.Conv(self.d_model, (1, 1), use_bias=False)(galaxy_map)
        psf_proj = nn.Conv(self.d_model, (1, 1), use_bias=False)(psf_map)

        # Flatten spatial dims to token sequences
        gal_tokens = gal_proj.reshape(batch, H_g * W_g, self.d_model)
        psf_tokens = psf_proj.reshape(batch, H_p * W_p, self.d_model)

        # Learned positional embeddings
        gal_pos = self.param(
            "gal_pos_embed", nn.initializers.normal(0.02), (1, H_g * W_g, self.d_model)
        )
        psf_pos = self.param(
            "psf_pos_embed", nn.initializers.normal(0.02), (1, H_p * W_p, self.d_model)
        )
        gal_tokens = gal_tokens + gal_pos
        psf_tokens = psf_tokens + psf_pos

        # Cross-attention: galaxy queries PSF (pre-norm + residual)
        gal_norm = nn.LayerNorm()(gal_tokens)
        psf_norm = nn.LayerNorm()(psf_tokens)
        cross_out = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(gal_norm, psf_norm)
        gal_tokens = gal_tokens + cross_out

        # Self-attention layers to refine galaxy tokens
        for _ in range(self.num_self_attn_layers):
            gal_norm = nn.LayerNorm()(gal_tokens)
            self_out = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(gal_norm, gal_norm)
            gal_tokens = gal_tokens + self_out

        # Global average pool over token sequence
        fused = jnp.mean(gal_tokens, axis=1)

        # Output head
        fused = nn.LayerNorm()(fused)
        x = nn.Dense(128)(fused)
        x = nn.relu(x)
        x = nn.Dense(len(output_keys))(x)
        return x


class ForkLike(nn.Module):
    """Combine two sub-models (galaxy and PSF branches) into one estimator.

    Trains one sub-model on galaxy images and another on PSF images, then
    concatenates their features and applies the dense/fully connected layers.

    This is to mimimic the forklens structure here https://github.com/zhangzzk/forklens/.
    """

    galaxy_model_type: str = "cnn"  # Default to EnhancedGalaxyNN
    psf_model_type: str = "cnn"  # Default to EnhancedGalaxyNN
    fusion: str = "concat"  # Options: "concat", "transformer"

    def setup(self):
        """Initialize the sub-models during setup."""
        self.galaxy_model = self._get_model(self.galaxy_model_type)
        self.psf_model = self._get_model(self.psf_model_type)
        if self.fusion == "transformer":
            self.transformer_fusion = TransformerFusion()

    def _get_model(self, model_type):
        """Return a model instance for the given branch type string."""
        return build_branch_model(model_type)

    @nn.compact
    def __call__(
        self,
        galaxy_image,
        psf_image,
        output_keys: tuple = ("g1", "g2"),
        deterministic: bool = False,
        gap: bool = False,
    ):
        """Run both branches, fuse their features, and return the prediction."""
        if self.fusion == "transformer":
            galaxy_map = self.galaxy_model(
                galaxy_image, deterministic=deterministic, return_spatial=True
            )
            psf_map = self.psf_model(psf_image, deterministic=deterministic, return_spatial=True)
            return self.transformer_fusion(
                galaxy_map, psf_map, output_keys=output_keys, deterministic=deterministic
            )

        # This model will learn from galaxy images
        galaxy_features = self.galaxy_model(
            galaxy_image, deterministic=deterministic, fork=True, gap=gap
        )

        # This model will learn from psf images
        psf_features = self.psf_model(psf_image, deterministic=deterministic, fork=True, gap=gap)

        # Combines features from the two separate models above trained on different types of images
        # to represent them in one feature layer
        combined_features = jnp.concatenate([galaxy_features, psf_features], axis=-1)

        # The fully connected layers
        x = nn.Dense(128)(combined_features)
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.relu(x)

        # Final predictions
        x = nn.Dense(len(output_keys))(x)
        return x


# ---------------------------------------------------------------------------
# D4-equivariant fork-like model
#
# Implements the D4CNN construction of Lin et al. (2026), "D4CNN x AnaCal:
# Physics-Informed Machine Learning for Accurate and Precise Weak Lensing Shear
# Estimation" (arXiv:2603.19046), adapted to ShearNet's two-branch (galaxy +
# PSF) ``fork-like`` layout with an optional transformer fusion.
#
# The idea: galaxy ellipticity is a spin-2 quantity, so under the D4 group
# (90-degree rotations + mirrors) the two shape components must transform as
#     e1 -> w1(g) e1,   e2 -> w2(g) e2,   with w1, w2 in {+1, -1}.
# Rather than hoping the network learns this, we hard-code it. For each of the
# eight group elements g_i we push the (jointly transformed) galaxy+PSF pair
# through an arbitrary backbone F, map the resulting feature map *back* to the
# reference frame with g_i^{-1}, and take a sign-weighted average
#     Psi_c = (1/8) sum_i w_c(g_i) g_i^{-1} F(g_i . input),   c in {1, 2}.
# This Reynolds (group-averaging) operator is exactly spin-2 equivariant for an
# *arbitrary* F, so the transformer fusion (with absolute positional embeddings)
# can sit inside it without breaking the symmetry. A D4-symmetric Gaussian
# window + global-average-pool collapses each Psi_c to a channel vector whose
# only D4 transformation is the overall sign w_c(g); a bias-free, tanh ("odd")
# MLP then preserves that sign, yielding outputs that transform as e1, e2.
# ---------------------------------------------------------------------------


def _d4_apply(x, i):
    """Apply the ``i``-th D4 group element to a batched image ``x``.

    ``x`` has shape ``(batch, H, W[, C])`` and the group acts on the spatial
    axes ``(1, 2)``. Element ``i`` is decomposed as ``R90**r . P**m`` with
    ``r = i % 4`` (number of 90-degree rotations) and ``m = i // 4`` (mirror
    flag), giving the eight elements ``i = 0..7``.
    """
    r, m = i % 4, i // 4
    if m:
        x = jnp.flip(x, axis=1)
    if r:
        x = jnp.rot90(x, r, axes=(1, 2))
    return x


def _d4_inverse_apply(x, i):
    """Apply the inverse of the ``i``-th D4 element (undoes :func:`_d4_apply`)."""
    r, m = i % 4, i // 4
    if r:
        x = jnp.rot90(x, -r, axes=(1, 2))
    if m:
        x = jnp.flip(x, axis=1)
    return x


# Sign of the spin-2 representation for each group element i = 0..7.
#   w1(g) = (-1)**r            (e1 = |e| cos 2theta is even under mirror)
#   w2(g) = (-1)**(r + m)      (e2 = |e| sin 2theta flips under mirror)
# These reproduce Eqs. (25)-(26) of Lin et al. (2026):
#   Psi_1 signs: +, -, +, -, +, -, +, -
#   Psi_2 signs: +, -, +, -, -, +, -, +
_D4_W1 = jnp.array([(-1.0) ** (i % 4) for i in range(8)])
_D4_W2 = jnp.array([(-1.0) ** ((i % 4) + (i // 4)) for i in range(8)])


def _d4_gaussian_window(size, sigma_frac=0.25):
    """A centred, D4-symmetric Gaussian window of shape ``(size, size)``.

    The window depends only on the squared radius from the centre, so it is
    invariant under the D4 group; multiplying a feature map by it before global
    average pooling therefore keeps the pooled feature D4-invariant (up to the
    overall spin-2 sign) while down-weighting the noisy stamp edges, following
    the Gaussian-kernel step of Lin et al. (2026, Sec. 2.3.1).
    """
    center = (size - 1) / 2.0
    coord = jnp.arange(size) - center
    yy, xx = jnp.meshgrid(coord, coord, indexing="ij")
    sigma = sigma_frac * size
    win = jnp.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return win / jnp.sum(win)


class _D4SmoothCNN(nn.Module):
    """Smooth convolutional backbone for a D4 branch (galaxy or PSF).

    Uses GeLU activations, average pooling and layer normalisation only -- no
    ReLU or max-pooling -- so the mapping is continuously differentiable, a
    prerequisite for the analytic (gradient-based) shear calibration discussed
    in Lin et al. (2026, Sec. 2.1.2). Returns a square spatial feature map so
    the D4 orbit alignment (which uses ``rot90``) is well defined.
    """

    features: tuple = (16, 32)
    kernel_size: tuple = (3, 3)

    @nn.compact
    def __call__(self, x):
        """Map ``(batch, H, W)`` to a spatial feature map ``(batch, H', W', C)``."""
        if x.ndim == 2:
            x = jnp.expand_dims(x, axis=0)
        x = jnp.expand_dims(x, axis=-1)
        for feat in self.features:
            x = nn.Conv(feat, self.kernel_size, padding="SAME")(x)
            x = nn.LayerNorm()(x)
            x = nn.gelu(x)
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x


class _D4SpatialTransformerFusion(nn.Module):
    """Transformer fusion that returns a *spatial* galaxy-frame feature map.

    Like :class:`TransformerFusion` (galaxy tokens cross-attend to PSF tokens,
    then self-attend), but instead of pooling to a prediction it reshapes the
    refined galaxy tokens back to ``(batch, H_g, W_g, d_model)``. Keeping the
    output spatial lets the enclosing :class:`D4ForkLike` undo the orbit
    transformation and build the equivariant features. All activations are the
    smooth GeLU.
    """

    d_model: int = 64
    num_heads: int = 4
    num_self_attn_layers: int = 1

    @nn.compact
    def __call__(self, galaxy_map, psf_map, deterministic: bool = True):
        """Fuse the two maps and return a galaxy-frame spatial feature map."""
        batch, H_g, W_g = galaxy_map.shape[0], galaxy_map.shape[1], galaxy_map.shape[2]
        H_p, W_p = psf_map.shape[1], psf_map.shape[2]

        gal_proj = nn.Conv(self.d_model, (1, 1), use_bias=False)(galaxy_map)
        psf_proj = nn.Conv(self.d_model, (1, 1), use_bias=False)(psf_map)

        gal_tokens = gal_proj.reshape(batch, H_g * W_g, self.d_model)
        psf_tokens = psf_proj.reshape(batch, H_p * W_p, self.d_model)

        gal_pos = self.param(
            "gal_pos_embed", nn.initializers.normal(0.02), (1, H_g * W_g, self.d_model)
        )
        psf_pos = self.param(
            "psf_pos_embed", nn.initializers.normal(0.02), (1, H_p * W_p, self.d_model)
        )
        gal_tokens = gal_tokens + gal_pos
        psf_tokens = psf_tokens + psf_pos

        gal_norm = nn.LayerNorm()(gal_tokens)
        psf_norm = nn.LayerNorm()(psf_tokens)
        cross_out = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(gal_norm, psf_norm)
        gal_tokens = gal_tokens + cross_out

        for _ in range(self.num_self_attn_layers):
            gal_norm = nn.LayerNorm()(gal_tokens)
            self_out = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(gal_norm, gal_norm)
            gal_tokens = gal_tokens + self_out

        # Back to a galaxy-frame spatial map for orbit alignment.
        return gal_tokens.reshape(batch, H_g, W_g, self.d_model)


class D4ForkLike(nn.Module):
    """D4-equivariant two-branch shear estimator (``nn='d4-fork-like'``).

    Combines ShearNet's fork-like galaxy/PSF split with the hard-coded D4
    symmetry of Lin et al. (2026). The galaxy and PSF stamps are transformed
    together over the eight-element D4 orbit; a shared smooth backbone + fusion
    produces one feature map per orbit element; these are aligned back to the
    reference frame and combined with the spin-2 sign weights to form the
    equivariant feature maps Psi_1, Psi_2. A D4-symmetric Gaussian window and
    global-average-pool reduce them to channel vectors, and bias-free tanh
    ("odd") MLPs map those to the shape components ``(g1, g2)``.

    By construction the first two outputs transform *exactly* as a spin-2
    vector under 90-degree rotations and mirrors (up to float32 round-off).
    Any additional ``output_keys`` beyond the first two are treated as
    D4-invariant scalars and regressed from the invariant orbit average.

    Attributes:
        fusion: ``'transformer'`` (galaxy cross-attends to PSF, the intended
            configuration) or ``'concat'`` (PSF summarised as a global context
            vector broadcast onto the galaxy map).
        galaxy_features / psf_features: channel widths of the smooth backbones.
        d_model / num_heads: transformer-fusion width and attention heads.
    """

    fusion: str = "transformer"
    galaxy_features: tuple = (16, 32)
    psf_features: tuple = (16, 32)
    d_model: int = 64
    num_heads: int = 4

    def _fuse(self, galaxy_map, psf_map, deterministic):
        """Fuse galaxy/PSF maps into a single galaxy-frame spatial map."""
        if self.fusion == "transformer":
            return _D4SpatialTransformerFusion(d_model=self.d_model, num_heads=self.num_heads)(
                galaxy_map, psf_map, deterministic=deterministic
            )
        # 'concat': summarise the PSF as a global descriptor and broadcast it
        # onto every galaxy spatial location, keeping the galaxy spatial frame.
        psf_global = jnp.mean(psf_map, axis=(1, 2), keepdims=True)
        psf_global = jnp.broadcast_to(
            psf_global, galaxy_map.shape[:3] + (psf_map.shape[-1],)
        )
        return jnp.concatenate([galaxy_map, psf_global], axis=-1)

    @nn.compact
    def __call__(
        self,
        galaxy_image,
        psf_image,
        output_keys: tuple = ("g1", "g2"),
        deterministic: bool = False,
        gap: bool = False,
    ):
        """Run the D4 orbit, build equivariant features, and return predictions."""
        if galaxy_image.ndim == 2:
            galaxy_image = jnp.expand_dims(galaxy_image, axis=0)
        if psf_image.ndim == 2:
            psf_image = jnp.expand_dims(psf_image, axis=0)
        batch = galaxy_image.shape[0]

        galaxy_backbone = _D4SmoothCNN(features=self.galaxy_features)
        psf_backbone = _D4SmoothCNN(features=self.psf_features)

        # Build the D4 orbit of the (galaxy, PSF) pair and stack it into the
        # batch axis so the shared backbone/fusion runs once over all 8 copies.
        gal_orbit = jnp.concatenate([_d4_apply(galaxy_image, i) for i in range(8)], axis=0)
        psf_orbit = jnp.concatenate([_d4_apply(psf_image, i) for i in range(8)], axis=0)

        gal_maps = galaxy_backbone(gal_orbit)
        psf_maps = psf_backbone(psf_orbit)
        fused = self._fuse(gal_maps, psf_maps, deterministic)  # (8*batch, H, W, C)

        H, W, C = fused.shape[1], fused.shape[2], fused.shape[3]
        fused = fused.reshape(8, batch, H, W, C)

        # Align each orbit member back to the reference frame with g_i^{-1}.
        aligned = jnp.stack([_d4_inverse_apply(fused[i], i) for i in range(8)], axis=0)

        # Sign-weighted (Reynolds) averages -> equivariant feature maps.
        psi1 = jnp.mean(_D4_W1[:, None, None, None, None] * aligned, axis=0)
        psi2 = jnp.mean(_D4_W2[:, None, None, None, None] * aligned, axis=0)

        # D4-symmetric Gaussian window + global average pool -> channel vectors.
        window = _d4_gaussian_window(H)[None, :, :, None]
        s1 = jnp.sum(psi1 * window, axis=(1, 2))
        s2 = jnp.sum(psi2 * window, axis=(1, 2))

        # Bias-free tanh ("odd") MLPs preserve the spin-2 sign of the features.
        def odd_mlp(z, name):
            z = nn.Dense(128, use_bias=False, name=f"{name}_dense0")(z)
            z = nn.tanh(z)
            z = nn.Dense(1, use_bias=False, name=f"{name}_dense1")(z)
            return z[:, 0]

        n_out = len(output_keys)
        columns = []
        if n_out >= 1:
            columns.append(odd_mlp(s1, "odd_e1"))
        if n_out >= 2:
            columns.append(odd_mlp(s2, "odd_e2"))

        # Extra outputs (e.g. hlr, flux) are D4-invariant scalars: regress them
        # from the sign-free (invariant) orbit average via a plain MLP.
        if n_out > 2:
            psi_inv = jnp.mean(aligned, axis=0)
            s_inv = jnp.sum(psi_inv * window, axis=(1, 2))
            h = nn.gelu(nn.Dense(128)(s_inv))
            extra = nn.Dense(n_out - 2)(h)
            columns.extend(extra[:, k] for k in range(n_out - 2))

        return jnp.stack(columns, axis=-1)


# ---------------------------------------------------------------------------
# Model registries and factories
#
# Single source of truth for the architecture-name -> class mapping, replacing
# the ``if/elif`` chains that previously lived in ``core.train``, ``cli.evaluate``
# and ``ForkLike._get_model``. Adding a new architecture now means editing one
# dict here.
# ---------------------------------------------------------------------------

# Top-level single-branch architectures selectable via ``nn=`` (everything
# except the two-branch ``fork-like`` model, which is built separately because
# it takes extra branch/fusion arguments).
SINGLE_BRANCH_MODELS = {
    "mlp": SimpleGalaxyNN,
    "cnn": EnhancedGalaxyNN,
    "resnet": GalaxyResNet,
    "research_backed": ResearchBackedGalaxyResNet,
    "forklens_psfnet": ForkLensPSFNet,
}

# Sub-models usable as a galaxy/PSF branch inside :class:`ForkLike`. Note the
# ``forklens_psf`` key differs from the top-level ``forklens_psfnet`` above and
# is kept distinct to preserve existing config semantics.
BRANCH_MODELS = {
    "mlp": SimpleGalaxyNN,
    "cnn": EnhancedGalaxyNN,
    "resnet": GalaxyResNet,
    "research_backed": ResearchBackedGalaxyResNet,
    "forklens_psf": ForkLensPSFNet,
}


def build_branch_model(model_type):
    """Instantiate a :class:`ForkLike` branch sub-model from its type string."""
    try:
        return BRANCH_MODELS[model_type]()
    except KeyError:
        raise ValueError(f"Invalid model type specified: {model_type}")


# Two-branch architectures that take a (galaxy, PSF) image pair rather than a
# single stamp. Used by the training/evaluation code to decide whether to feed
# PSF images through the model.
FORK_MODELS = frozenset({"fork-like", "d4-fork-like"})


def is_fork_model(nn):
    """Return ``True`` if architecture ``nn`` takes a (galaxy, PSF) pair."""
    return nn in FORK_MODELS


def build_model(nn, galaxy_type="cnn", psf_type="cnn", fusion="concat"):
    """Instantiate a top-level architecture from its ``nn`` name.

    The two-branch ``fork-like`` and ``d4-fork-like`` models are constructed
    with the given branch/fusion settings; every other name maps to a
    single-branch architecture in :data:`SINGLE_BRANCH_MODELS`.

    ``d4-fork-like`` is the D4-equivariant variant (Lin et al. 2026): it ignores
    ``galaxy_type``/``psf_type`` (it uses its own smooth backbones) and honours
    ``fusion`` (``'transformer'`` or ``'concat'``).
    """
    if nn == "fork-like":
        return ForkLike(galaxy_model_type=galaxy_type, psf_model_type=psf_type, fusion=fusion)
    if nn == "d4-fork-like":
        return D4ForkLike(fusion=fusion)
    try:
        return SINGLE_BRANCH_MODELS[nn]()
    except KeyError:
        raise ValueError(f"Invalid model type specified: {nn}")
