#!/usr/bin/env python3
"""
Test script for Linear Sum Assignment dataset generation.
This script verifies that the shuffle_lsa function preserves the optimal assignment.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def shuffle_lsa(cost_matrix: np.ndarray, assignment: np.ndarray):
    """
    Shuffle a Linear Sum Assignment instance similar to shuffle_sudoku.
    """
    n = cost_matrix.shape[0]
    
    # Generate random row and column permutations
    row_perm = np.random.permutation(n)
    col_perm = np.random.permutation(n)
    
    # Create inverse permutations for assignment update
    inv_row_perm = np.argsort(row_perm)
    inv_col_perm = np.argsort(col_perm)
    
    # Shuffle the cost matrix
    shuffled_cost = cost_matrix[row_perm][:, col_perm]
    
    # Update assignment
    shuffled_assignment = np.zeros(n, dtype=np.int32)
    for old_row in range(n):
        old_col = assignment[old_row]
        new_row = inv_row_perm[old_row]
        new_col = inv_col_perm[old_col]
        shuffled_assignment[new_row] = new_col
    
    return shuffled_cost, shuffled_assignment


def generate_lsa_instance(n: int, max_value: int):
    """Generate a random Linear Sum Assignment instance."""
    cost_matrix = np.random.randint(1, max_value + 1, size=(n, n))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignment = np.zeros(n, dtype=np.int32)
    assignment[row_ind] = col_ind
    return cost_matrix, assignment


def calculate_cost(cost_matrix: np.ndarray, assignment: np.ndarray):
    """Calculate the total cost of an assignment."""
    return sum(cost_matrix[i, assignment[i]] for i in range(len(assignment)))


def test_shuffle_preserves_optimality():
    """Test that shuffling preserves the optimality of the assignment."""
    print("Testing shuffle_lsa function...")
    
    np.random.seed(42)
    n = 9
    max_value = 100
    num_tests = 10
    
    for test_idx in range(num_tests):
        # Generate original instance
        cost_matrix, assignment = generate_lsa_instance(n, max_value)
        original_cost = calculate_cost(cost_matrix, assignment)
        
        # Shuffle
        shuffled_cost, shuffled_assignment = shuffle_lsa(cost_matrix, assignment)
        shuffled_cost_value = calculate_cost(shuffled_cost, shuffled_assignment)
        
        # Verify the assignment is still optimal for the shuffled matrix
        row_ind, col_ind = linear_sum_assignment(shuffled_cost)
        optimal_shuffled_assignment = np.zeros(n, dtype=np.int32)
        optimal_shuffled_assignment[row_ind] = col_ind
        optimal_cost = calculate_cost(shuffled_cost, optimal_shuffled_assignment)
        
        print(f"Test {test_idx + 1}:")
        print(f"  Original cost: {original_cost}")
        print(f"  Shuffled assignment cost: {shuffled_cost_value}")
        print(f"  Optimal shuffled cost: {optimal_cost}")
        
        # The costs should match
        if shuffled_cost_value == optimal_cost:
            print(f"  ✓ PASS: Shuffled assignment is optimal")
        else:
            print(f"  ✗ FAIL: Shuffled assignment is suboptimal!")
            print(f"    Original matrix:\n{cost_matrix}")
            print(f"    Original assignment: {assignment}")
            print(f"    Shuffled matrix:\n{shuffled_cost}")
            print(f"    Shuffled assignment: {shuffled_assignment}")
            print(f"    Optimal assignment: {optimal_shuffled_assignment}")
            return False
        
        # Verify the assignment is a valid permutation
        if len(np.unique(shuffled_assignment)) != n:
            print(f"  ✗ FAIL: Shuffled assignment is not a valid permutation!")
            return False
        
        print()
    
    print("All tests passed! ✓")
    return True


def demonstrate_usage():
    """Demonstrate how the LSA dataset works."""
    print("\n" + "="*60)
    print("DEMONSTRATION: Linear Sum Assignment Dataset")
    print("="*60 + "\n")
    
    np.random.seed(123)
    
    # Generate a small example
    print("1. Generate a 5x5 cost matrix:")
    cost_matrix, assignment = generate_lsa_instance(5, 20)
    print(f"\nCost Matrix:\n{cost_matrix}")
    print(f"\nOptimal Assignment: {assignment}")
    print(f"(Row i is assigned to column {assignment[0]}, {assignment[1]}, ...)")
    print(f"\nTotal Cost: {calculate_cost(cost_matrix, assignment)}")
    
    # Show shuffled version
    print("\n2. Generate shuffled version:")
    shuffled_cost, shuffled_assignment = shuffle_lsa(cost_matrix, assignment)
    print(f"\nShuffled Cost Matrix:\n{shuffled_cost}")
    print(f"\nShuffled Optimal Assignment: {shuffled_assignment}")
    print(f"\nTotal Cost: {calculate_cost(shuffled_cost, shuffled_assignment)}")
    
    print("\n3. Key observations:")
    print("   - The shuffled matrix is different from the original")
    print("   - The assignment is still optimal for the shuffled matrix")
    print("   - The total cost is preserved (this is the key property!)")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # Run tests
    if test_shuffle_preserves_optimality():
        # Show demonstration
        demonstrate_usage()
