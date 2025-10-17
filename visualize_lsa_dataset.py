#!/usr/bin/env python3
"""
Visualize examples from the Linear Sum Assignment dataset.
"""

import numpy as np
import json
import os
import sys


def load_dataset(dataset_path, split="train"):
    """Load LSA dataset from disk."""
    
    # Load metadata
    with open(os.path.join(dataset_path, split, "dataset.json"), "r") as f:
        metadata = json.load(f)
    
    # Load data
    inputs = np.load(os.path.join(dataset_path, split, "all__inputs.npy"), mmap_mode='r')
    labels = np.load(os.path.join(dataset_path, split, "all__labels.npy"), mmap_mode='r')
    puzzle_indices = np.load(os.path.join(dataset_path, split, "all__puzzle_indices.npy"))
    
    return {
        'metadata': metadata,
        'inputs': inputs,
        'labels': labels,
        'puzzle_indices': puzzle_indices
    }


def visualize_example(cost_matrix, assignment, example_id):
    """Print a single LSA example nicely formatted."""
    n = len(assignment)
    
    print(f"\n{'='*60}")
    print(f"Example {example_id}")
    print(f"{'='*60}\n")
    
    print("Cost Matrix:")
    print(cost_matrix)
    print()
    
    print(f"Optimal Assignment: {assignment}")
    print("\nAssignment breakdown:")
    total_cost = 0
    for i, j in enumerate(assignment):
        cost = cost_matrix[i, j]
        total_cost += cost
        print(f"  Row {i} → Column {j} (cost: {cost})")
    
    print(f"\nTotal Cost: {total_cost}")
    print()


def verify_assignment(cost_matrix, assignment):
    """Verify that an assignment is a valid permutation and calculate cost."""
    n = len(assignment)
    
    # Check it's a valid permutation
    if len(set(assignment)) != n:
        return False, -1, "Not a valid permutation"
    
    if not all(0 <= x < n for x in assignment):
        return False, -1, "Assignment values out of range"
    
    # Calculate cost
    total_cost = sum(cost_matrix[i, assignment[i]] for i in range(n))
    
    return True, total_cost, "Valid"


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_lsa_dataset.py <dataset_path> [num_examples]")
        print("Example: python visualize_lsa_dataset.py data/lsa-demo 3")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    num_examples = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    data = load_dataset(dataset_path, split="train")
    
    # Print metadata
    print("\nDataset Metadata:")
    print(f"  Sequence length: {data['metadata']['seq_len']}")
    print(f"  Vocab size: {data['metadata']['vocab_size']}")
    print(f"  Total samples: {len(data['inputs'])}")
    print(f"  Total puzzles: {data['metadata']['total_puzzles']}")
    print(f"  Mean examples per puzzle: {data['metadata']['mean_puzzle_examples']}")
    
    # Infer matrix size from sequence length
    matrix_size = int(np.sqrt(data['metadata']['seq_len']))
    print(f"  Matrix size: {matrix_size}×{matrix_size}")
    
    # Show examples
    print(f"\nShowing first {num_examples} examples:")
    
    for i in range(min(num_examples, len(data['inputs']))):
        # Reshape input to matrix
        cost_matrix = data['inputs'][i].reshape(matrix_size, matrix_size)
        assignment = data['labels'][i]
        
        visualize_example(cost_matrix, assignment, i)
        
        # Verify
        valid, cost, msg = verify_assignment(cost_matrix, assignment)
        if valid:
            print(f"✓ Assignment is valid (cost: {cost})")
        else:
            print(f"✗ Assignment is invalid: {msg}")
    
    # Statistics
    print(f"\n{'='*60}")
    print("Dataset Statistics")
    print(f"{'='*60}\n")
    
    all_costs = []
    for i in range(len(data['inputs'])):
        cost_matrix = data['inputs'][i].reshape(matrix_size, matrix_size)
        assignment = data['labels'][i]
        _, cost, _ = verify_assignment(cost_matrix, assignment)
        all_costs.append(cost)
    
    all_costs = np.array(all_costs)
    print(f"Total Cost Statistics:")
    print(f"  Mean: {np.mean(all_costs):.2f}")
    print(f"  Std:  {np.std(all_costs):.2f}")
    print(f"  Min:  {np.min(all_costs)}")
    print(f"  Max:  {np.max(all_costs)}")
    print(f"  Median: {np.median(all_costs):.2f}")
    
    # Check for duplicates (to see augmentation working)
    print(f"\nAugmentation Check:")
    unique_costs = len(np.unique(all_costs))
    print(f"  Unique cost values: {unique_costs} out of {len(all_costs)} samples")
    if unique_costs < len(all_costs):
        print(f"  → Augmentations are working (same puzzles with different permutations)")
    else:
        print(f"  → No augmentations detected or all costs are unique")


if __name__ == "__main__":
    main()
