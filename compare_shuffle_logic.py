#!/usr/bin/env python3
"""
Compare the shuffling logic between Sudoku and Linear Sum Assignment datasets.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def compare_shuffle_logic():
    """Compare how Sudoku and LSA datasets handle shuffling/augmentation."""
    
    print("="*80)
    print("COMPARISON: Sudoku vs Linear Sum Assignment Shuffling")
    print("="*80)
    
    print("\n" + "─"*80)
    print("1. SUDOKU DATASET SHUFFLING")
    print("─"*80)
    
    print("""
    The shuffle_sudoku function creates augmented versions by:
    
    1. Digit Mapping:
       - Random permutation of digits 1-9 (0/blank unchanged)
       - Example: 1→7, 2→3, 3→1, etc.
    
    2. Transposition:
       - Randomly decide to transpose the board (50% chance)
    
    3. Band/Stack Permutations:
       - Shuffle the 3 bands (groups of 3 rows)
       - Within each band, shuffle the 3 rows
       - Similarly for columns (stacks)
    
    Key Property: A valid Sudoku remains a valid Sudoku with the same solution
                  structure after these transformations
    """)
    
    print("─"*80)
    print("2. LINEAR SUM ASSIGNMENT SHUFFLING")
    print("─"*80)
    
    print("""
    The shuffle_lsa function creates augmented versions by:
    
    1. Row Permutation:
       - Random permutation of all rows
       - Example: row [0,1,2,...,8] → [3,7,1,...]
    
    2. Column Permutation:
       - Random permutation of all columns
       - Example: col [0,1,2,...,8] → [5,2,8,...]
    
    3. Assignment Update:
       - Adjust the assignment to match the permuted matrix
       - If old: row i → col j, new: row inv_row[i] → col inv_col[j]
    
    Key Property: The optimal assignment remains optimal with the same total cost
                  after these transformations
    """)
    
    print("\n" + "─"*80)
    print("3. SIDE-BY-SIDE EXAMPLE")
    print("─"*80)
    
    # LSA example
    np.random.seed(42)
    cost_matrix = np.random.randint(1, 20, size=(5, 5))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignment = np.zeros(5, dtype=int)
    assignment[row_ind] = col_ind
    original_cost = sum(cost_matrix[i, assignment[i]] for i in range(5))
    
    print(f"\nLSA Original Cost Matrix (5×5):")
    print(cost_matrix)
    print(f"\nOptimal Assignment: {assignment}")
    print(f"Total Cost: {original_cost}")
    
    # Apply shuffle
    row_perm = np.array([2, 0, 4, 1, 3])
    col_perm = np.array([1, 3, 0, 4, 2])
    shuffled_cost = cost_matrix[row_perm][:, col_perm]
    
    inv_row_perm = np.argsort(row_perm)
    inv_col_perm = np.argsort(col_perm)
    shuffled_assignment = np.zeros(5, dtype=int)
    for old_row in range(5):
        old_col = assignment[old_row]
        new_row = inv_row_perm[old_row]
        new_col = inv_col_perm[old_col]
        shuffled_assignment[new_row] = new_col
    
    shuffled_cost_value = sum(shuffled_cost[i, shuffled_assignment[i]] for i in range(5))
    
    print(f"\nLSA Shuffled Cost Matrix (rows: {list(row_perm)}, cols: {list(col_perm)}):")
    print(shuffled_cost)
    print(f"\nShuffled Assignment: {shuffled_assignment}")
    print(f"Total Cost: {shuffled_cost_value}")
    print(f"\n✓ Cost preserved: {original_cost} = {shuffled_cost_value}")
    
    print("\n" + "─"*80)
    print("4. KEY SIMILARITIES & DIFFERENCES")
    print("─"*80)
    
    print("""
    SIMILARITIES:
    ✓ Both create different-looking but equivalent problem instances
    ✓ Both preserve the fundamental solution property (validity/optimality)
    ✓ Both use permutations as the core transformation
    ✓ Both integrate with the same dataset infrastructure
    
    DIFFERENCES:
    ✗ Sudoku: Complex structure-aware shuffling (bands, stacks, digit mapping)
    ✗ LSA: Simple independent row/column permutations
    ✗ Sudoku: Preserves constraint satisfaction
    ✗ LSA: Preserves optimization objective value
    ✗ Sudoku: Output same size as input (81→81)
    ✗ LSA: Output smaller than input (81→9)
    """)
    
    print("\n" + "─"*80)
    print("5. USAGE IN TRAINING")
    print("─"*80)
    
    print("""
    Both datasets use the same augmentation strategy:
    
    For each original puzzle:
      - Keep original version (aug_idx = 0)
      - Generate N augmented versions (aug_idx = 1..N)
      - All versions share the same 'group_id' but different 'puzzle_id'
    
    This increases dataset diversity without changing problem statistics.
    
    Example with num_aug=2:
      100 original puzzles → 300 total samples (100 × 3)
    """)
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    print("""
    The LSA dataset successfully adapts the Sudoku shuffling philosophy to a
    different problem type (optimization vs constraint satisfaction). Both use
    permutation-based augmentation to create diverse training examples while
    preserving the problem's mathematical properties.
    
    The implementation follows the same code structure, making it easy to
    integrate LSA into the existing puzzle dataset training pipeline.
    """)


if __name__ == "__main__":
    compare_shuffle_logic()
