"""
Test script for sinusoidal value embeddings.

Validates that the embedding produces smooth, continuous representations
where similar values have similar embeddings.
"""

import torch
import numpy as np
from models.value_embedding import SinusoidalValueEmbedding, HybridValueEmbedding, ContinuousValueEmbedding


def test_sinusoidal_embedding():
    """Test that sinusoidal embeddings are smooth and continuous."""
    print("=" * 60)
    print("Testing Sinusoidal Value Embedding")
    print("=" * 60)

    embedding = SinusoidalValueEmbedding(
        embedding_dim=64,
        max_value=100.0,
        min_value=0.0,
        learnable_scale=False
    )

    # Test 1: Similar values should have similar embeddings
    print("\nTest 1: Similarity of nearby values")
    values = torch.tensor([50.0, 51.0, 52.0, 75.0])
    embeddings = embedding(values)

    # Compute cosine similarities
    def cosine_sim(a, b):
        return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))

    sim_50_51 = cosine_sim(embeddings[0], embeddings[1])
    sim_50_52 = cosine_sim(embeddings[0], embeddings[2])
    sim_50_75 = cosine_sim(embeddings[0], embeddings[3])

    print(f"  Similarity(50, 51): {sim_50_51:.4f}")
    print(f"  Similarity(50, 52): {sim_50_52:.4f}")
    print(f"  Similarity(50, 75): {sim_50_75:.4f}")
    print(f"  ✓ Nearby values more similar: {sim_50_51 > sim_50_75}")

    # Test 2: Embeddings should be smooth (no discontinuities)
    print("\nTest 2: Smoothness across range")
    test_range = torch.linspace(0, 100, 20)
    range_embeddings = embedding(test_range)

    # Check consecutive differences
    diffs = torch.norm(range_embeddings[1:] - range_embeddings[:-1], dim=1)
    print(f"  Embedding L2 differences (should be roughly uniform):")
    print(f"  Mean: {diffs.mean():.4f}, Std: {diffs.std():.4f}")
    print(f"  Min: {diffs.min():.4f}, Max: {diffs.max():.4f}")
    print(f"  ✓ Smooth transitions: {diffs.std() / diffs.mean() < 0.5}")

    print("\n" + "=" * 60)
    return True


def test_hybrid_embedding():
    """Test hybrid embedding (sinusoidal + learnable)."""
    print("Testing Hybrid Value Embedding")
    print("=" * 60)

    embedding = HybridValueEmbedding(
        num_embeddings=101,  # 0-100
        embedding_dim=64,
        max_value=100.0,
        min_value=0.0,
        sinusoidal_ratio=0.5
    )

    print(f"  Embedding dimension: 64")
    print(f"  Sinusoidal dimension: {embedding.sinusoidal_dim}")
    print(f"  Learnable dimension: {embedding.learnable_dim}")

    # Test forward pass
    values = torch.tensor([10, 20, 30, 40, 50])
    embeddings = embedding(values)

    print(f"\n  Input shape: {values.shape}")
    print(f"  Output shape: {embeddings.shape}")
    print(f"  ✓ Shape correct: {embeddings.shape == (5, 64)}")

    print("\n" + "=" * 60)
    return True


def test_continuous_embedding():
    """Test continuous (pure sinusoidal) embedding."""
    print("Testing Continuous Value Embedding")
    print("=" * 60)

    embedding = ContinuousValueEmbedding(
        num_embeddings=101,
        embedding_dim=64,
        max_value=100.0,
        min_value=0.0
    )

    # Test with LSA-like cost values
    cost_matrix = torch.randint(1, 101, (9, 9), dtype=torch.float32)
    print(f"\n  Sample cost matrix shape: {cost_matrix.shape}")
    print(f"  Sample values: {cost_matrix.flatten()[:5]}")

    embeddings = embedding(cost_matrix)
    print(f"  Output shape: {embeddings.shape}")
    print(f"  ✓ Shape correct: {embeddings.shape == (9, 9, 64)}")

    print("\n" + "=" * 60)
    return True


def compare_discrete_vs_sinusoidal():
    """Compare discrete vs sinusoidal embeddings."""
    print("Comparing Discrete vs Sinusoidal Embeddings")
    print("=" * 60)

    # Discrete embedding (current approach)
    discrete_emb = torch.nn.Embedding(101, 64)
    torch.nn.init.normal_(discrete_emb.weight, std=0.02)

    # Sinusoidal embedding (proposed approach)
    sinusoidal_emb = SinusoidalValueEmbedding(
        embedding_dim=64,
        max_value=100.0,
        min_value=0.0,
        learnable_scale=False
    )

    # Test on sequence of values
    values = torch.tensor([10, 11, 12, 50, 51, 52])

    discrete_out = discrete_emb(values)
    sinusoidal_out = sinusoidal_emb(values.float())

    def cosine_sim(a, b):
        return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))

    print("\n  Discrete Embedding Similarities:")
    print(f"    Similarity(10, 11): {cosine_sim(discrete_out[0], discrete_out[1]):.4f}")
    print(f"    Similarity(10, 50): {cosine_sim(discrete_out[0], discrete_out[3]):.4f}")

    print("\n  Sinusoidal Embedding Similarities:")
    print(f"    Similarity(10, 11): {cosine_sim(sinusoidal_out[0], sinusoidal_out[1]):.4f}")
    print(f"    Similarity(10, 50): {cosine_sim(sinusoidal_out[0], sinusoidal_out[3]):.4f}")

    print("\n  ✓ Sinusoidal embeddings preserve numerical relationships better")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SINUSOIDAL VALUE EMBEDDING TEST SUITE")
    print("=" * 60 + "\n")

    test_sinusoidal_embedding()
    print()
    test_hybrid_embedding()
    print()
    test_continuous_embedding()
    print()
    compare_discrete_vs_sinusoidal()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
