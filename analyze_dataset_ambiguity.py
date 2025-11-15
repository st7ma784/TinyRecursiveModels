"""
Analyze the LSA dataset to check how often multiple optimal solutions might exist.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
import os

def check_multiple_solutions(cost_matrix):
    """
    Check if an LSA problem likely has multiple optimal solutions.
    We check for duplicate rows or columns which guarantee multiple solutions.
    """
    n = cost_matrix.shape[0]

    # Check for duplicate rows
    for i in range(n):
        for j in range(i+1, n):
            if np.array_equal(cost_matrix[i], cost_matrix[j]):
                return True, "duplicate_rows"

    # Check for duplicate columns
    for i in range(n):
        for j in range(i+1, n):
            if np.array_equal(cost_matrix[:, i], cost_matrix[:, j]):
                return True, "duplicate_columns"

    return False, None


def analyze_dataset(data_dir, max_samples=1000):
    """Analyze the dataset for potential multiple solutions."""

    inputs_file = os.path.join(data_dir, "all__inputs.npy")
    if not os.path.exists(inputs_file):
        print(f"Dataset not found at {inputs_file}")
        return

    inputs = np.load(inputs_file)
    print(f"Loaded {len(inputs)} samples from {data_dir}")

    matrix_size = int(np.sqrt(inputs.shape[1]))
    print(f"Matrix size: {matrix_size}x{matrix_size}")

    ambiguous_count = 0
    duplicate_rows_count = 0
    duplicate_cols_count = 0

    samples_to_check = min(max_samples, len(inputs))

    for i in range(samples_to_check):
        cost_matrix = inputs[i].reshape(matrix_size, matrix_size)
        has_duplicates, dup_type = check_multiple_solutions(cost_matrix)

        if has_duplicates:
            ambiguous_count += 1
            if dup_type == "duplicate_rows":
                duplicate_rows_count += 1
            elif dup_type == "duplicate_columns":
                duplicate_cols_count += 1

    print(f"\n{'='*60}")
    print(f"Analysis Results (first {samples_to_check} samples):")
    print(f"{'='*60}")
    print(f"Samples with duplicate rows: {duplicate_rows_count} ({100*duplicate_rows_count/samples_to_check:.2f}%)")
    print(f"Samples with duplicate columns: {duplicate_cols_count} ({100*duplicate_cols_count/samples_to_check:.2f}%)")
    print(f"Total potentially ambiguous: {ambiguous_count} ({100*ambiguous_count/samples_to_check:.2f}%)")
    print(f"\nNote: This is a lower bound - there may be other cases with")
    print(f"multiple optimal solutions that don't have exact duplicate rows/cols.")


if __name__ == "__main__":
    # Check training data
    print("Analyzing TRAINING data:")
    analyze_dataset("data/lsa-9x9-10k/train", max_samples=10000)

    print("\n" + "="*60 + "\n")

    # Check test data
    print("Analyzing TEST data:")
    analyze_dataset("data/lsa-9x9-10k/test", max_samples=1000)
