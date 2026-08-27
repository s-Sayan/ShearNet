"""Flax neural-network architectures for galaxy shear estimation."""

import jax
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

    Attributes:
        dropout: Spatial-dropout rate (Srivastava et al., JMLR 2014) applied to
            the final feature map, just before it is returned as spatial features
            or flattened. ``0.0`` (default) inserts no dropout layer at all, so
            the parameter tree and every existing (inference / evaluation) call
            path are byte-for-byte unchanged. A positive rate is the primary
            anti-overfit lever for this high-capacity backbone; it is stochastic
            only when ``deterministic=False`` (training) and is an exact identity
            at inference (``deterministic=True``), so any equivariance guarantee of
            an enclosing model is preserved for all evaluated quantities.
    """

    dropout: float = 0.0

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

        # OPTIONAL REGULARIZATION on the final feature map.
        # CITATION: "Dropout: A Simple Way to Prevent Neural Networks from
        # Overfitting" (Srivastava et al., JMLR 2014).
        # The layer is only *created* when a positive rate is requested, so the
        # default (dropout=0.0) leaves the module tree — and every existing
        # inference/eval call path, which never supplies a 'dropout' rng — exactly
        # as before. When active it draws a mask only under deterministic=False
        # (training) and is the identity under deterministic=True (inference).
        if self.dropout and self.dropout > 0.0:
            x = nn.Dropout(rate=self.dropout, deterministic=deterministic)(x)

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


def _blur_pool(x):
    """Anti-aliased 2x downsample: a fixed [1,2,1]x[1,2,1]/16 blur, then 2x2 average pooling.

    The kernel is separable, so it is applied as two 1-D passes; it is fixed
    rather than learned, and D4-symmetric. Low-pass filtering before
    subsampling suppresses the frequencies the coarser grid cannot represent,
    which is what makes strided sampling stable under sub-pixel shifts
    (Zhang, ICML 2019). Odd sizes lose the last row/column to the pooling, so
    53 -> 26 -> 13.
    """
    k = jnp.array([0.25, 0.5, 0.25], dtype=x.dtype)
    for axis in (1, 2):
        pad = [(0, 0)] * x.ndim
        pad[axis] = (1, 1)
        xp = jnp.pad(x, pad)
        n = x.shape[axis]
        x = sum(k[i] * jax.lax.slice_in_dim(xp, i, i + n, axis=axis) for i in range(3))
    return nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))


class _SmoothResBlock(nn.Module):
    """Pre-normalised residual block ``x + alpha * u2`` (ShearNet-D4 report, Sec. 4.1).

    Two LayerNorm/GELU/3x3-convolution stages feed a scaled residual. LayerNorm
    normalises over channels at each spatial location, so -- unlike batch
    normalisation -- a faint galaxy's representation never depends on the S/N
    distribution of its minibatch neighbours, and training and inference are the
    same function. The small ``alpha`` keeps the block near the identity at
    initialisation, which is what you want when the sought distortion is far
    smaller than the intrinsic morphology.
    """

    features: int
    alpha: float = 0.1

    @nn.compact
    def __call__(self, x):
        """Apply the residual block and return a map with ``features`` channels."""
        skip = x
        if x.shape[-1] != self.features:
            skip = nn.Conv(self.features, (1, 1), use_bias=False)(x)
        u = nn.Conv(self.features, (3, 3), padding="SAME")(nn.gelu(nn.LayerNorm()(x)))
        u = nn.Conv(self.features, (3, 3), padding="SAME")(nn.gelu(nn.LayerNorm()(u)))
        return skip + self.alpha * u


class _SmoothMultiScaleBlock(nn.Module):
    """Dilated multiscale context block (ShearNet-D4 report, Sec. 5.1).

    Three parallel 3x3 convolutions at dilations (1, 2, 4) give effective
    kernels of 3, 5 and 9 feature-grid pixels; their concatenation is projected
    back to ``features`` and added residually. Dilation buys the receptive field
    without a further resolution cut (Yu & Koltun, ICLR 2016), so at 13x13 the
    three paths separate compact cores, intermediate morphology and
    nearly stamp-wide context.
    """

    features: int = 64
    scale_features: int = 32
    dilations: tuple = (1, 2, 4)
    alpha: float = 0.1

    @nn.compact
    def __call__(self, x):
        """Apply the multiscale context block and return a map of the same shape."""
        z = nn.gelu(nn.LayerNorm()(x))
        paths = [
            nn.Conv(self.scale_features, (3, 3), padding="SAME", kernel_dilation=(d, d))(z)
            for d in self.dilations
        ]
        return x + self.alpha * nn.Conv(self.features, (1, 1))(jnp.concatenate(paths, axis=-1))


class _ShearNetD4Backbone(nn.Module):
    """Smooth residual backbone of the ShearNet-D4 report (Secs. 4-6).

    Three stages at widths ``features`` separated by :func:`_blur_pool`, so a
    53x53 stamp reaches the 13x13 fusion resolution as
    ``53x53x32 -> 26x26x48 -> 13x13x64``. Each stage is a 1x1 transition
    followed by ``depths[stage]`` :class:`_SmoothResBlock`s; the galaxy branch
    additionally ends in a :class:`_SmoothMultiScaleBlock`.

    Everything here is convolution, LayerNorm, GELU and fixed linear pooling --
    no ReLU kink, no max pooling, no batch statistic -- so the image derivatives
    the response losses differentiate through stay well behaved.

    Attributes:
        features: channel width of each stage.
        depths: number of residual blocks per stage. The report's galaxy branch
            is ``(2, 2, 1)`` and its lighter PSF branch is ``(1, 1, 1)``.
        multiscale: append the dilated context block (galaxy branch only).
    """

    features: tuple = (32, 48, 64)
    depths: tuple = (2, 2, 1)
    multiscale: bool = True

    @nn.compact
    def __call__(self, x):
        """Map ``(batch, H, W)`` to a spatial feature map ``(batch, H/4, W/4, C)``."""
        if x.ndim == 2:
            x = jnp.expand_dims(x, axis=0)
        x = jnp.expand_dims(x, axis=-1)
        # Stem: 3x3 convolution, LayerNorm, GELU.
        x = nn.gelu(nn.LayerNorm()(nn.Conv(self.features[0], (3, 3), padding="SAME")(x)))
        for stage, (feat, depth) in enumerate(zip(self.features, self.depths)):
            if stage:
                x = _blur_pool(x)
                x = nn.Conv(feat, (1, 1))(x)
            for _ in range(depth):
                x = _SmoothResBlock(feat)(x)
        if self.multiscale:
            x = _SmoothMultiScaleBlock(self.features[-1])(x)
        return x


class _D4SpatialTransformerFusion(nn.Module):
    """Transformer fusion that returns a *spatial* galaxy-frame feature map.

    Like :class:`TransformerFusion` (galaxy tokens cross-attend to PSF tokens,
    then self-attend), but instead of pooling to a prediction it reshapes the
    refined galaxy tokens back to ``(batch, H_g, W_g, d_model)``. Keeping the
    output spatial lets the enclosing :class:`D4ForkLike` undo the orbit
    transformation and build the equivariant features. All activations are the
    smooth GeLU.

    ``ffn_dim`` adds the pre-normalised feed-forward sublayer after each
    attention sublayer, as in the original transformer block (Vaswani et al.
    2017) and required by the ShearNet-D4 report (Secs. 7.2-7.3). ``0``, the
    default, omits them entirely, leaving the attention-only block -- and its
    parameter tree -- exactly as it was.
    """

    d_model: int = 64
    num_heads: int = 4
    num_self_attn_layers: int = 1
    ffn_dim: int = 0

    def _ffn(self, tokens):
        """Pre-normalised residual feed-forward sublayer."""
        h = nn.gelu(nn.Dense(self.ffn_dim)(nn.LayerNorm()(tokens)))
        return tokens + nn.Dense(self.d_model)(h)

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
        if self.ffn_dim:
            gal_tokens = self._ffn(gal_tokens)

        for _ in range(self.num_self_attn_layers):
            gal_norm = nn.LayerNorm()(gal_tokens)
            self_out = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(gal_norm, gal_norm)
            gal_tokens = gal_tokens + self_out
            if self.ffn_dim:
                gal_tokens = self._ffn(gal_tokens)

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

    Because the Reynolds average is exactly equivariant for an *arbitrary*
    backbone F, the galaxy and PSF branches are pluggable: ``galaxy_branch`` /
    ``psf_branch`` select which spatial backbone runs inside the orbit. Every
    choice yields a D4-equivariant model; they differ only in feature-extraction
    capacity (and smoothness -- see the note on ``d4cnn`` below).

    Attributes:
        fusion: ``'transformer'`` (galaxy cross-attends to PSF, the intended
            configuration) or ``'concat'`` (PSF summarised as a global context
            vector broadcast onto the galaxy map).
        galaxy_branch / psf_branch: spatial backbone for each branch, one of
            :data:`D4_BRANCH_BACKBONES` (``'d4cnn'`` default). ``'d4cnn'`` is the
            smooth (GeLU / avg-pool) :class:`_D4SmoothCNN`, which keeps the map
            continuously differentiable -- a prerequisite for the analytic
            gradient-based calibration of Lin et al. (2026, Sec 2.1.2).
            ``'shearnet-d4'`` is the equally smooth but far deeper
            :class:`_ShearNetD4Backbone`, which reaches the fusion stage at
            13x13 instead of 1x1 and is the branch the ShearNet-D4 report
            specifies; see ``design`` below, since choosing it also changes the
            fusion block and the heads.
            ``'research_backed'`` and ``'forklens_psf'`` are higher-capacity but
            use ReLU / strided convs, so they stay equivariant but forgo that
            smoothness. Any backbone must return a *square* spatial map so the
            ``rot90`` orbit alignment is well defined.
        galaxy_features / psf_features: channel widths of the ``d4cnn`` backbone
            (ignored by the other branches, which fix their own widths).
        d_model / num_heads: transformer-fusion width and attention heads.
        dropout: spatial-dropout rate forwarded to a ``research_backed`` branch
            (ignored by ``d4cnn`` / ``forklens_psf``, which take no dropout). It
            is stochastic only during training (``deterministic=False``) and an
            exact identity at inference, so the model's spin-2 equivariance holds
            for every evaluated quantity. ``0.0`` (default) inserts no dropout.
        design: which specification the fusion block and the output heads follow.
            ``'d4cnn'`` (default) is the attention-only fusion and single-hidden-
            layer heads described above. ``'shearnet-d4'`` selects the ShearNet-D4
            report's versions of the same three pieces -- feed-forward sublayers
            in the fusion block (Secs. 7.2-7.3), two-hidden-layer odd shape heads
            (Sec. 10.1), and one output layer per invariant scalar (Sec. 10.2).
            The report specifies branches, fusion and heads as a single
            architecture, so ``build_model`` sets this from the branch name
            rather than exposing it as an independent config key.
    """

    fusion: str = "transformer"
    galaxy_branch: str = "d4cnn"
    psf_branch: str = "d4cnn"
    galaxy_features: tuple = (16, 32)
    psf_features: tuple = (16, 32)
    d_model: int = 64
    num_heads: int = 4
    head: str = "gap"  # 'gap' (fixed Gaussian window) | 'attention'
    num_pool_heads: int = 4  # number of attention pooling maps (head='attention')
    dropout: float = 0.0  # spatial-dropout rate for a 'research_backed' branch
    design: str = "d4cnn"  # 'd4cnn' | 'shearnet-d4' -- fusion and head layout

    def _branch_map(self, branch, features, x, deterministic, psf=False):
        """Run the chosen spatial backbone over the orbit and return its map.

        Any backbone returning a square spatial feature map is valid; the
        enclosing Reynolds average makes the whole model exactly spin-2
        equivariant regardless of the backbone's internal structure. ``psf``
        selects the lighter variant of any backbone that has one.
        """
        if branch == "d4cnn":
            return _D4SmoothCNN(features=features)(x)
        if branch == "shearnet-d4":
            # The report gives the galaxy branch two residual blocks per stage
            # plus the dilated context block, and the PSF branch one block per
            # stage and no context block: a PSF kernel needs enough capacity for
            # anisotropy and wings, not a morphology hierarchy.
            depths = (1, 1, 1) if psf else (2, 2, 1)
            return _ShearNetD4Backbone(depths=depths, multiscale=not psf)(x)
        if branch == "research_backed":
            # ``dropout`` regularizes only this high-capacity backbone; it is
            # stochastic under training (deterministic=False) and an exact
            # identity at inference (deterministic=True), so the Reynolds average
            # stays exactly spin-2 equivariant for every evaluated quantity.
            return ResearchBackedGalaxyResNet(dropout=self.dropout)(
                x, deterministic=deterministic, return_spatial=True
            )
        if branch == "forklens_psf":
            return ForkLensPSFNet()(x, deterministic=deterministic, return_spatial=True)
        raise ValueError(
            f"Unknown D4 branch {branch!r}; choose from " f"{sorted(D4_BRANCH_BACKBONES)}"
        )

    def _fuse(self, galaxy_map, psf_map, deterministic):
        """Fuse galaxy/PSF maps into a single galaxy-frame spatial map."""
        if self.fusion == "transformer":
            return _D4SpatialTransformerFusion(
                d_model=self.d_model,
                num_heads=self.num_heads,
                # Sec. 7.2: a width-256 GELU feed-forward after each attention
                # sublayer. 0 keeps the attention-only block.
                ffn_dim=256 if self.design == "shearnet-d4" else 0,
            )(galaxy_map, psf_map, deterministic=deterministic)
        # 'concat': summarise the PSF as a global descriptor and broadcast it
        # onto every galaxy spatial location, keeping the galaxy spatial frame.
        psf_global = jnp.mean(psf_map, axis=(1, 2), keepdims=True)
        psf_global = jnp.broadcast_to(psf_global, galaxy_map.shape[:3] + (psf_map.shape[-1],))
        return jnp.concatenate([galaxy_map, psf_global], axis=-1)

    @nn.compact
    def __call__(
        self,
        galaxy_image,
        psf_image,
        output_keys: tuple = ("g1", "g2"),
        deterministic: bool = False,
        gap: bool = False,
        capture_attention: bool = False,
    ):
        """Run the D4 orbit, build equivariant features, and return predictions.

        When ``capture_attention`` is true, the four spatial pooling maps are
        exposed in Flax's ``intermediates`` collection under
        ``"pool_attention"``.  The default is false so model initialisation and
        ordinary prediction keep exactly the same variable tree and return
        value as before.
        """
        if galaxy_image.ndim == 2:
            galaxy_image = jnp.expand_dims(galaxy_image, axis=0)
        if psf_image.ndim == 2:
            psf_image = jnp.expand_dims(psf_image, axis=0)
        batch = galaxy_image.shape[0]

        # Build the D4 orbit of the (galaxy, PSF) pair and stack it into the
        # batch axis so each branch backbone/fusion runs once over all 8 copies.
        gal_orbit = jnp.concatenate([_d4_apply(galaxy_image, i) for i in range(8)], axis=0)
        psf_orbit = jnp.concatenate([_d4_apply(psf_image, i) for i in range(8)], axis=0)

        # Galaxy branch is created first, then PSF (preserves 'd4cnn' checkpoint
        # param naming). Any square-map backbone stays exactly spin-2 equivariant
        # once wrapped in the Reynolds average below.
        gal_maps = self._branch_map(
            self.galaxy_branch, self.galaxy_features, gal_orbit, deterministic
        )
        psf_maps = self._branch_map(
            self.psf_branch, self.psf_features, psf_orbit, deterministic, psf=True
        )
        fused = self._fuse(gal_maps, psf_maps, deterministic)  # (8*batch, H, W, C)

        H, W, C = fused.shape[1], fused.shape[2], fused.shape[3]
        fused = fused.reshape(8, batch, H, W, C)

        # Align each orbit member back to the reference frame with g_i^{-1}.
        aligned = jnp.stack([_d4_inverse_apply(fused[i], i) for i in range(8)], axis=0)

        # Sign-weighted (Reynolds) averages -> equivariant feature maps.
        psi1 = jnp.mean(_D4_W1[:, None, None, None, None] * aligned, axis=0)
        psi2 = jnp.mean(_D4_W2[:, None, None, None, None] * aligned, axis=0)

        # Pool each equivariant map to a channel vector that carries only the
        # spin-2 sign w_c: a D4-consistent spatial sum kills the rotation part and
        # keeps the sign. ``psi_inv`` is the sign-free (trivial-rep) context map --
        # it transforms purely spatially, so any weighting derived from it rotates
        # WITH psi1/psi2 and the pooled vector stays exactly spin-2 equivariant.
        psi_inv = jnp.mean(aligned, axis=0)

        if self.head == "attention":
            # Content-adaptive pooling: K learnable attention maps (from the
            # sign-free context) replace the single fixed Gaussian window. The head
            # then sees K spatial regions and a K*C-wide vector instead of
            # collapsing everything through one radial window -- lifting the
            # information bottleneck WITHOUT breaking equivariance (the weights
            # rotate with the maps and a spatial sum is rotation-invariant).
            K = self.num_pool_heads
            ctx = nn.gelu(nn.Dense(self.d_model, name="pool_ctx")(psi_inv))
            logits = nn.Dense(K, name="pool_logits")(ctx)  # (B, H, W, K)
            attn = jax.nn.softmax(logits.reshape(batch, H * W, K), axis=1)
            attn = attn.reshape(batch, H, W, K)
            if capture_attention:
                self.sow("intermediates", "pool_attention", attn)

            def _pool(psi):
                # (B,H,W,C) weighted by (B,H,W,K) -> (B, K*C)
                return jnp.einsum("bhwc,bhwk->bkc", psi, attn).reshape(batch, -1)

        else:
            # 'gap': original single fixed D4-symmetric Gaussian window + GAP.
            window = _d4_gaussian_window(H)[None, :, :, None]

            def _pool(psi):
                return jnp.sum(psi * window, axis=(1, 2))

        s1, s2 = _pool(psi1), _pool(psi2)

        # Bias-free tanh ("odd") MLPs preserve the spin-2 sign of the features:
        # a bias-free linear map and tanh are both odd, so f(-s) = -f(s) and the
        # sign carried by psi1/psi2 survives the head. The report (Sec. 10.1)
        # uses two hidden layers, 256 -> 128 -> 128 -> 1.
        hidden = (128, 128) if self.design == "shearnet-d4" else (128,)

        def odd_mlp(z, name):
            for i, width in enumerate(hidden):
                z = nn.tanh(nn.Dense(width, use_bias=False, name=f"{name}_dense{i}")(z))
            z = nn.Dense(1, use_bias=False, name=f"{name}_dense{len(hidden)}")(z)
            return z[:, 0]

        n_out = len(output_keys)
        columns = []
        if n_out >= 1:
            columns.append(odd_mlp(s1, "odd_e1"))
        if n_out >= 2:
            columns.append(odd_mlp(s2, "odd_e2"))

        # Extra outputs (e.g. hlr, flux) are D4-invariant scalars: regress them
        # from the sign-free (invariant) pooled context via a plain MLP.
        if n_out > 2:
            s_inv = _pool(psi_inv)
            h = nn.gelu(nn.Dense(128)(s_inv))
            if self.design == "shearnet-d4":
                # Sec. 10.2: one final linear layer per scalar. log-size and
                # log-flux have different noise and dynamic ranges, so they do
                # not share an undifferentiated multi-output layer.
                columns.extend(
                    nn.Dense(1, name=f"scalar_{key}")(h)[:, 0] for key in output_keys[2:]
                )
            else:
                extra = nn.Dense(n_out - 2)(h)
                columns.extend(extra[:, k] for k in range(n_out - 2))

        return jnp.stack(columns, axis=-1)


def attention_pool_diagnostics(attention, eps=1e-12):
    """Summarise a batch of attention-pooling maps without discarding the maps.

    Args:
        attention: ``(batch, height, width, heads)`` probability maps captured
            from :class:`D4ForkLike`'s ``"pool_attention"`` intermediate.
        eps: numerical floor used by logarithms and normalisations.

    Returns:
        A dictionary containing per-head normalised spatial entropy, the full
        mean cosine-similarity matrix, mean/max off-diagonal similarity, and an
        effective head rank in ``[1, heads]``.  Identical pooling maps have
        similarity one and effective rank one, making head collapse explicit.
    """
    attention = jnp.asarray(attention)
    if attention.ndim != 4:
        raise ValueError(
            "attention pooling maps must have shape (batch, height, width, heads), "
            f"got {attention.shape}"
        )
    batch, height, width, heads = attention.shape
    spatial = height * width
    flat = attention.reshape(batch, spatial, heads)

    entropy = -jnp.sum(flat * jnp.log(jnp.maximum(flat, eps)), axis=1)
    entropy = jnp.mean(entropy / jnp.log(jnp.asarray(spatial, attention.dtype)), axis=0)

    unit = flat / jnp.maximum(jnp.linalg.norm(flat, axis=1, keepdims=True), eps)
    similarity = jnp.mean(jnp.einsum("bnh,bnk->bhk", unit, unit), axis=0)
    offdiag = ~jnp.eye(heads, dtype=bool)
    offdiag_values = similarity[offdiag]
    mean_similarity = jnp.mean(offdiag_values) if heads > 1 else jnp.asarray(0.0)
    max_similarity = jnp.max(offdiag_values) if heads > 1 else jnp.asarray(0.0)

    eigenvalues = jnp.maximum(jnp.linalg.eigvalsh(similarity), 0.0)
    spectrum = eigenvalues / jnp.maximum(jnp.sum(eigenvalues), eps)
    effective_rank = jnp.exp(
        -jnp.sum(spectrum * jnp.log(jnp.maximum(spectrum, eps)))
    )
    return {
        "entropy": entropy,
        "similarity": similarity,
        "mean_similarity": mean_similarity,
        "max_similarity": max_similarity,
        "effective_rank": effective_rank,
    }


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


# Spatial backbones usable as a galaxy/PSF branch inside :class:`D4ForkLike`.
# Each must return a SQUARE spatial feature map (the orbit alignment uses
# ``rot90``); the enclosing Reynolds average then makes the whole model exactly
# spin-2 equivariant regardless of the backbone. ``d4cnn`` is the smooth default;
# ``shearnet-d4`` is the smooth residual backbone of the ShearNet-D4 report; the
# others are higher-capacity but non-smooth (see :class:`D4ForkLike`).
D4_BRANCH_BACKBONES = {
    "d4cnn": _D4SmoothCNN,
    "shearnet-d4": _ShearNetD4Backbone,
    "research_backed": ResearchBackedGalaxyResNet,
    "forklens_psf": ForkLensPSFNet,
}


# Two-branch architectures that take a (galaxy, PSF) image pair rather than a
# single stamp. Used by the training/evaluation code to decide whether to feed
# PSF images through the model.
FORK_MODELS = frozenset({"fork-like", "d4-fork-like"})


def is_fork_model(nn):
    """Return ``True`` if architecture ``nn`` takes a (galaxy, PSF) pair."""
    return nn in FORK_MODELS


def build_model(
    nn,
    galaxy_type="cnn",
    psf_type="cnn",
    fusion="concat",
    head="gap",
    dropout=0.0,
    branch_features=None,
):
    """Instantiate a top-level architecture from its ``nn`` name.

    The two-branch ``fork-like`` and ``d4-fork-like`` models are constructed
    with the given branch/fusion settings; every other name maps to a
    single-branch architecture in :data:`SINGLE_BRANCH_MODELS`.

    ``d4-fork-like`` is the D4-equivariant variant (Lin et al. 2026). It honours
    ``fusion`` (``'transformer'`` or ``'concat'``) and now maps
    ``galaxy_type``/``psf_type`` onto its pluggable D4 branches
    (:data:`D4_BRANCH_BACKBONES`, e.g. ``'d4cnn'`` / ``'research_backed'`` /
    ``'forklens_psf'``); ``None`` falls back to the smooth ``'d4cnn'`` backbone,
    which reproduces the original behaviour.

    ``dropout`` (default ``0.0``) sets the spatial-dropout rate of the
    ``research_backed`` backbone -- an anti-overfit lever for that high-capacity
    branch, wherever it appears (a ``d4-fork-like`` branch or the single-branch
    ``research_backed`` model). It is ignored by every other architecture. At
    ``0.0`` no dropout layer is created, so the parameter tree and all existing
    call paths are unchanged; a checkpoint trained with dropout loads and
    evaluates identically when rebuilt at the default, because dropout adds no
    parameters and is an exact identity at inference.
    """
    if nn == "fork-like":
        return ForkLike(galaxy_model_type=galaxy_type, psf_model_type=psf_type, fusion=fusion)
    if nn == "d4-fork-like":

        def _d4_branch(t):
            # ``build_model``'s generic default is 'cnn'; treat that (and None)
            # as the smooth 'd4cnn' backbone so bare d4-fork-like construction
            # reproduces the original behaviour. Explicit D4 branch names pass
            # through; a typo passes through too and raises in ``_branch_map``.
            return "d4cnn" if t in (None, "cnn") else t

        # branch_features sizes the d4cnn backbone. Lin et al. (2026) use five
        # layers at base width 32; the default here is the two-layer (16, 32)
        # proof-of-concept, so a baseline meant to reproduce their numbers has
        # to say so rather than inherit a smaller network silently.
        widths = tuple(branch_features) if branch_features else (16, 32)
        galaxy_branch = _d4_branch(galaxy_type)
        return D4ForkLike(
            fusion=fusion,
            galaxy_branch=galaxy_branch,
            psf_branch=_d4_branch(psf_type),
            galaxy_features=widths,
            psf_features=widths,
            head=head or "gap",
            dropout=dropout or 0.0,
            # The ShearNet-D4 report specifies branches, fusion and heads as one
            # architecture, so selecting its galaxy backbone also selects its
            # fusion feed-forward sublayers and its head depths. Every other
            # branch keeps the existing layout.
            design="shearnet-d4" if galaxy_branch == "shearnet-d4" else "d4cnn",
        )
    try:
        model_cls = SINGLE_BRANCH_MODELS[nn]
    except KeyError:
        raise ValueError(f"Invalid model type specified: {nn}")
    # Only the research_backed backbone consumes a dropout rate; a positive value
    # on any other single-branch model is silently ignored (they take no such
    # argument). dropout=0.0 constructs every model exactly as before.
    if dropout and nn == "research_backed":
        return model_cls(dropout=dropout)
    return model_cls()
