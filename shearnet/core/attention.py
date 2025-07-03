"""
Added for attention implementation
"""

import flax.linen as nn
import jax.numpy as jnp

class SpatialAttention(nn.Module):
	"""Spatial Attention implementation"""
	
	@nn.compact
	def __call__(self, x):
		avg_pool = jnp.mean(x, axis=-1, keepdims=True)
		max_pool = jnp.max(x, axis=-1, keepdims=True)

		pooled = jnp.concatenate([avg_pool, max_pool], axis=-1)
		attention = nn.Conv(1, (7,7), padding='SAME')(pooled)
		attention = nn.sigmoid(attention)

		return x * attention
