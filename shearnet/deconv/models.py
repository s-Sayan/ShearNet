"""PSF deconvolution networks for neural metacalibration.

This module implements U-Net based architectures for learning PSF deconvolution
directly from data, replacing traditional Fourier-space methods.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Optional, Tuple
import numpy as np


class ConvBlock(nn.Module):
    """Convolutional block with batch norm and activation."""
    features: int
    kernel_size: Tuple[int, int] = (3, 3)
    use_batch_norm: bool = True
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        x = nn.Conv(self.features, self.kernel_size, padding='SAME')(x)
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        
        x = nn.Conv(self.features, self.kernel_size, padding='SAME')(x)
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        
        #if self.dropout_rate > 0:
            #x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        
        return x


class UNetEncoder(nn.Module):
    """Encoder part of U-Net."""
    features: Sequence[int]
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        skip_connections = []
        
        for i, feat in enumerate(self.features):
            x = ConvBlock(feat, dropout_rate=self.dropout_rate)(x, deterministic)
            skip_connections.append(x)
            
            if i < len(self.features) - 1:  # Don't pool on last layer
                x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        return x, skip_connections


class UNetDecoder(nn.Module):
    """Decoder part of U-Net with skip connections - handles odd dimensions."""
    features: Sequence[int]
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x, skip_connections, deterministic: bool = False):
        # Reverse the features for upsampling
        features_reversed = list(reversed(self.features[:-1]))
        skip_reversed = list(reversed(skip_connections[:-1]))
        
        for i, (feat, skip) in enumerate(zip(features_reversed, skip_reversed)):
            # Upsample
            x = nn.ConvTranspose(feat, (2, 2), strides=(2, 2))(x)
            
            # Handle dimension mismatch by cropping or padding
            skip_h, skip_w = skip.shape[1], skip.shape[2]
            x_h, x_w = x.shape[1], x.shape[2]
            
            # Step 1: Crop x if it's larger than skip in any dimension
            if x_h > skip_h:
                crop_h = (x_h - skip_h) // 2
                x = x[:, crop_h:crop_h+skip_h, :, :]
            if x_w > skip_w:
                crop_w = (x_w - skip_w) // 2
                x = x[:, :, crop_w:crop_w+skip_w, :]
            
            # Step 2: Pad x if it's smaller than skip in any dimension
            x_h_current, x_w_current = x.shape[1], x.shape[2]
            if x_h_current < skip_h or x_w_current < skip_w:
                pad_h = skip_h - x_h_current
                pad_w = skip_w - x_w_current
                pad_h_before = pad_h // 2
                pad_h_after = pad_h - pad_h_before
                pad_w_before = pad_w // 2
                pad_w_after = pad_w - pad_w_before
                x = jnp.pad(x, ((0, 0), (pad_h_before, pad_h_after), 
                           (pad_w_before, pad_w_after), (0, 0)), mode='constant')
            
            # Concatenate skip connection
            x = jnp.concatenate([x, skip], axis=-1)
            
            # Apply conv block
            x = ConvBlock(feat, dropout_rate=self.dropout_rate)(x, deterministic)
        
        return x


class AttentionGate(nn.Module):
    """Attention gate for focusing on important features."""
    features: int
    
    @nn.compact
    def __call__(self, g, x):
        """
        g: gating signal from decoder
        x: skip connection from encoder
        """
        # Match dimensions
        theta_x = nn.Conv(self.features, (1, 1), padding='SAME')(x)
        phi_g = nn.Conv(self.features, (1, 1), padding='SAME')(g)
        
        # Add and activate
        combined = nn.relu(theta_x + phi_g)
        
        # Attention coefficients
        psi = nn.Conv(1, (1, 1), padding='SAME')(combined)
        alpha = nn.sigmoid(psi)
        
        # Apply attention
        return x * alpha


class PSFDeconvolutionNet(nn.Module):
    """U-Net based PSF deconvolution network.
    
    This network learns to deconvolve PSF effects from galaxy images
    using a U-Net architecture with optional attention mechanisms.
    """
    features: Sequence[int] = (32, 64, 128, 256)
    use_attention: bool = True
    dropout_rate: float = 0.1
    output_channels: int = 1
    
    @nn.compact
    def __call__(self, galaxy_image, psf_image, training: bool = False):
        """
        Args:
            galaxy_image: Observed galaxy image [batch, height, width, channels]
            psf_image: PSF image [batch, height, width, channels]
            training: Whether in training mode (affects dropout/batch norm)
        
        Returns:
            Deconvolved galaxy image
        """
        deterministic = not training
        
        # Ensure proper dimensions
        if galaxy_image.ndim == 3:
            galaxy_image = jnp.expand_dims(galaxy_image, axis=-1)
        if psf_image.ndim == 3:
            psf_image = jnp.expand_dims(psf_image, axis=-1)
        
        # Concatenate galaxy and PSF as input
        x = jnp.concatenate([galaxy_image, psf_image], axis=-1)
        
        # Initial convolution to process concatenated input
        x = nn.Conv(self.features[0], (3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        
        # Encoder path
        encoder = UNetEncoder(self.features, self.dropout_rate)
        bottleneck, skip_connections = encoder(x, deterministic)
        
        # Bottleneck processing with optional attention
        if self.use_attention:
            # Self-attention in bottleneck
            b, h, w, c = bottleneck.shape
            bottleneck_flat = bottleneck.reshape(b, h*w, c)
            bottleneck_attended = nn.SelfAttention(num_heads=4)(bottleneck_flat)
            bottleneck = bottleneck_attended.reshape(b, h, w, c)
        
        # Decoder path
        decoder = UNetDecoder(self.features, self.dropout_rate)
        x = decoder(bottleneck, skip_connections, deterministic)
        
        # Final convolution to output
        x = nn.Conv(self.output_channels, (1, 1), padding='SAME')(x)
        
        # Residual connection with input galaxy image
        deconvolved = galaxy_image[..., :self.output_channels] + x
        
        return deconvolved


class SimplePSFDeconvNet(nn.Module):
    """Simpler deconvolution network for faster training."""
    features: int = 64
    num_layers: int = 4
    
    @nn.compact
    def __call__(self, galaxy_image, psf_image, training: bool = False):
        """Simple convolutional network for PSF deconvolution."""
        deterministic = not training
        
        # Ensure proper dimensions
        if galaxy_image.ndim == 3:
            galaxy_image = jnp.expand_dims(galaxy_image, axis=-1)
        if psf_image.ndim == 3:
            psf_image = jnp.expand_dims(psf_image, axis=-1)
        
        # Concatenate inputs
        x = jnp.concatenate([galaxy_image, psf_image], axis=-1)
        
        # Apply conv layers
        for i in range(self.num_layers):
            x = nn.Conv(self.features, (3, 3), padding='SAME')(x)
            x = nn.BatchNorm(use_running_average=True)(x)
            x = nn.relu(x)
        
        # Output layer
        x = nn.Conv(1, (3, 3), padding='SAME')(x)
        
        # Add residual
        return galaxy_image[..., :1] + x


def create_deconv_net(config):
    """Create deconvolution network from configuration."""
    arch = config.get('metacal.deconv_network.architecture', 'unet')
    
    if arch == 'unet':
        return PSFDeconvolutionNet(
            features=config.get('metacal.deconv_network.features', [32, 64, 128, 256]),
            use_attention=config.get('metacal.deconv_network.use_attention', True),
            dropout_rate=config.get('metacal.deconv_network.dropout_rate', 0.1)
        )
    elif arch == 'simple':
        return SimplePSFDeconvNet(
            features=config.get('metacal.deconv_network.features', 64),
            num_layers=config.get('metacal.deconv_network.num_layers', 4)
        )
    else:
        raise ValueError(f"Unknown deconvolution architecture: {arch}")