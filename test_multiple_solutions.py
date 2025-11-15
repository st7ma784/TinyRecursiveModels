"""
Test if LSA problems can have multiple optimal solutions.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

def test_multiple_solutions():
    """Test a simple case where multiple optimal solutions exist."""

    # Simple cost matrix with symmetry - multiple optimal solutions possible
    # Example: if two rows have identical costs, swapping them gives same total cost
    cost_matrix = np.array([
        [1, 2, 3],
        [1, 2, 3],  # Same as row 0
        [3, 2, 1]
    ])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignment1 = np.zeros(3, dtype=int)
    assignment1[row_ind] = col_ind

    print("Cost matrix:")
    print(cost_matrix)
    print(f"\nOptimal assignment 1: {assignment1}")

    # Calculate cost
    cost1 = sum(cost_matrix[i, assignment1[i]] for i in range(3))
    print(f"Cost of assignment 1: {cost1}")

    # Try alternative assignment (swap rows 0 and 1)
    assignment2 = np.array([assignment1[1], assignment1[0], assignment1[2]])
    cost2 = sum(cost_matrix[i, assignment2[i]] for i in range(3))

    print(f"\nAlternative assignment 2: {assignment2}")
    print(f"Cost of assignment 2: {cost2}")

    if cost1 == cost2 and not np.array_equal(assignment1, assignment2):
        print("\n✓ CONFIRMED: Multiple optimal solutions exist!")
        print("This means the model could find a different but equally valid solution")
        print("and still be marked as INCORRECT in the current loss computation.")
        return True

    # Try with a more general case - uniform costs
    print("\n" + "="*60)
    print("Testing with uniform cost matrix:")
    cost_matrix_uniform = np.ones((3, 3))
    print(cost_matrix_uniform)
    print("\nWith uniform costs, ALL permutations are optimal!")
    print("Total possible assignments: 3! = 6")
    print("But ground truth is only ONE of them.")

    return False

if __name__ == "__main__":
    test_multiple_solutions()
