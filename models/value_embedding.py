"""
Continuous Value Embeddings using Sinusoidal Encoding

For tasks with ordinal/continuous input values (like LSA cost matrices),
discrete embeddings fail to capture the numerical relationships between values.
This module provides sinusoidal embeddings similar to positional encodings.
"""

import torch
import torch.nn as nn
import math


class SinusoidalValueEmbedding(nn.Module):
    """
    Sinusoidal embedding for continuous/ordinal values.

    Similar to positional encodings, but applied to input values.
    Each value is encoded using multiple sine and cosine waves of different frequencies.

    Args:
        embedding_dim: Dimension of the output embedding
        max_value: Maximum value to encode (used for normalization)
        min_value: Minimum value to encode (default 0)
        learnable_scale: If True, makes the frequency scaling learnable
    """

    def __init__(
        self,
        embedding_dim: int,
        max_value: float = 100.0,
        min_value: float = 0.0,
        learnable_scale: bool = True,
        base: float = 10000.0
    ):
        super().__init__()

        assert embedding_dim % 2 == 0, "embedding_dim must be even"

        self.embedding_dim = embedding_dim
        self.max_value = max_value
        self.min_value = min_value
        self.base = base

        # Create frequency bands (half for sin, half for cos)
        # Similar to positional encoding: 1 / (base^(2i/d))
        dim_half = embedding_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, dim_half, dtype=torch.float32) / dim_half))

        # Register as buffer (not a parameter, but moves with model to GPU)
        self.register_buffer('inv_freq', inv_freq)

        # Optional learnable scaling (allows model to adjust frequency ranges)
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('scale', torch.ones(1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """
        Encode values using sinusoidal embedding.

        Args:
            values: Tensor of shape [...] containing values to embed
                    Values should be in range [min_value, max_value]

        Returns:
            Embeddings of shape [..., embedding_dim]
        """
        # Normalize values to [0, 1] range
        normalized = (values.float() - self.min_value) / (self.max_value - self.min_value)
        normalized = normalized.clamp(0, 1)

        # Apply learnable scaling
        normalized = normalized * self.scale

        # Compute sinusoidal encoding
        # Shape: [..., 1] @ [dim_half] -> [..., dim_half]
        angles = normalized.unsqueeze(-1) * self.inv_freq.unsqueeze(0)

        # Concatenate sin and cos
        embeddings = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        return embeddings


class HybridValueEmbedding(nn.Module):
    """
    Hybrid embedding combining sinusoidal encoding with learnable embeddings.

    This provides both:
    1. Inductive bias from sinusoidal encoding (smooth, continuous)
    2. Flexibility from learnable embeddings (task-specific patterns)

    Args:
        num_embeddings: Number of discrete values (vocab size)
        embedding_dim: Dimension of the output embedding
        max_value: Maximum value for sinusoidal encoding
        min_value: Minimum value for sinusoidal encoding
        sinusoidal_ratio: Ratio of dimensions for sinusoidal vs learned (default 0.5)
        init_std: Standard deviation for learned embedding initialization
        cast_to: Data type to cast to
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        max_value: float,
        min_value: float = 0.0,
        sinusoidal_ratio: float = 0.5,
        init_std: float = 1.0,
        cast_to: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.cast_to = cast_to
        self.embedding_dim = embedding_dim

        # Split dimensions between sinusoidal and learnable
        # Ensure both are even for sinusoidal
        sinusoidal_dim = int(embedding_dim * sinusoidal_ratio)
        sinusoidal_dim = sinusoidal_dim - (sinusoidal_dim % 2)  # Make even
        learnable_dim = embedding_dim - sinusoidal_dim

        self.sinusoidal_dim = sinusoidal_dim
        self.learnable_dim = learnable_dim

        # Sinusoidal component
        if sinusoidal_dim > 0:
            self.sinusoidal = SinusoidalValueEmbedding(
                embedding_dim=sinusoidal_dim,
                max_value=max_value,
                min_value=min_value,
                learnable_scale=True
            )

        # Learnable component
        if learnable_dim > 0:
            from models.common import trunc_normal_init_
            self.learned_embedding = nn.Parameter(
                trunc_normal_init_(
                    torch.empty((num_embeddings, learnable_dim)),
                    std=init_std
                )
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Integer tensor of shape [...] containing value indices

        Returns:
            Embeddings of shape [..., embedding_dim]
        """
        components = []

        # Sinusoidal component (based on value magnitude)
        if self.sinusoidal_dim > 0:
            # Use the input values directly for sinusoidal encoding
            sin_emb = self.sinusoidal(input.float())
            components.append(sin_emb)

        # Learnable component (discrete lookup)
        if self.learnable_dim > 0:
            learned_emb = torch.nn.functional.embedding(
                input.to(torch.int32),
                self.learned_embedding
            )
            components.append(learned_emb)

        # Concatenate and cast
        embedding = torch.cat(components, dim=-1)
        return embedding.to(self.cast_to)


# For backward compatibility / easy swapping
class ContinuousValueEmbedding(nn.Module):
    """
    Pure sinusoidal embedding for continuous values.
    Drop-in replacement for nn.Embedding for continuous/ordinal data.
    """

    def __init__(
        self,
        num_embeddings: int,  # For compatibility, represents max value range
        embedding_dim: int,
        init_std: float = 1.0,  # Unused, for compatibility
        cast_to: torch.dtype = torch.float32,
        max_value: float = 100.0,
        min_value: float = 0.0,
    ):
        super().__init__()

        self.cast_to = cast_to
        self.sinusoidal = SinusoidalValueEmbedding(
            embedding_dim=embedding_dim,
            max_value=max_value,
            min_value=min_value,
            learnable_scale=True
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Tensor of shape [...] containing values to embed

        Returns:
            Embeddings of shape [..., embedding_dim]
        """
        return self.sinusoidal(input).to(self.cast_to)
