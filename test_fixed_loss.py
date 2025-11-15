"""
Test the fixed loss function to verify it correctly handles alternative optimal solutions.
"""
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

# Import the fixed functions
import sys
sys.path.insert(0, '/data/TinyRecursiveModels')
from models.losses_fixed import (
    binary_tokens_to_assignment,
    calculate_assignment_cost,
    is_valid_permutation,
    cost_based_seq_is_correct
)


def create_lsa_tokens(assignment, matrix_size):
    """Convert assignment vector to token sequence (as done in dataset generation)."""
    binary_matrix = np.zeros((matrix_size, matrix_size), dtype=np.int32)
    for i, j in enumerate(assignment):
        binary_matrix[i, j] = 1
    # Flatten and add 1 to make it 1-indexed (values are 1 or 2)
    return torch.tensor(binary_matrix.flatten() + 1)


def create_fake_logits(tokens, vocab_size):
    """Create logits that would result in the given tokens when argmaxed."""
    # Shape: (seq_len, vocab_size)
    logits = torch.zeros(len(tokens), vocab_size)
    for i, token in enumerate(tokens):
        logits[i, token] = 10.0  # High value for the target token
    return logits


def test_cost_based_correctness():
    """Test that cost-based correctness works for alternative optimal solutions."""

    print("="*60)
    print("Testing Cost-Based Correctness Fix")
    print("="*60)

    # Create a cost matrix with multiple optimal solutions
    cost_matrix = np.array([
        [1, 2, 3],
        [1, 2, 3],  # Same as row 0 - allows swapping
        [3, 2, 1]
    ])

    matrix_size = 3

    # Ground truth assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    gt_assignment = np.zeros(matrix_size, dtype=int)
    gt_assignment[row_ind] = col_ind

    # Alternative optimal assignment (swap rows 0 and 1)
    alt_assignment = np.array([gt_assignment[1], gt_assignment[0], gt_assignment[2]])

    # Convert to tokens
    gt_tokens = create_lsa_tokens(gt_assignment, matrix_size)
    alt_tokens = create_lsa_tokens(alt_assignment, matrix_size)

    # Create inputs (cost matrix flattened)
    inputs = torch.tensor(cost_matrix.flatten()).unsqueeze(0)  # (1, seq_len)
    labels = gt_tokens.unsqueeze(0)  # (1, seq_len)

    # Create logits for alternative solution
    vocab_size = 102  # PAD + values 1..100 for inputs + 1-2 for labels
    pred_logits = create_fake_logits(alt_tokens, vocab_size).unsqueeze(0)  # (1, seq_len, vocab_size)

    print(f"\nCost matrix:")
    print(cost_matrix)
    print(f"\nGT assignment: {gt_assignment}")
    print(f"Alt assignment: {alt_assignment}")
    print(f"\nGT tokens: {gt_tokens.numpy()}")
    print(f"Alt tokens: {alt_tokens.numpy()}")

    # Calculate costs
    gt_cost = sum(cost_matrix[i, gt_assignment[i]] for i in range(matrix_size))
    alt_cost = sum(cost_matrix[i, alt_assignment[i]] for i in range(matrix_size))
    print(f"\nGT cost: {gt_cost}")
    print(f"Alt cost: {alt_cost}")
    print(f"Costs are equal: {gt_cost == alt_cost}")

    # Test the fixed function
    seq_is_correct = cost_based_seq_is_correct(
        inputs=inputs,
        labels=labels,
        pred_logits=pred_logits,
        matrix_size=matrix_size
    )

    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
    print(f"seq_is_correct (cost-based): {seq_is_correct.item()}")

    # Compare with original approach (exact token match)
    pred_tokens = torch.argmax(pred_logits, dim=-1)
    exact_match = (pred_tokens == labels).all()
    print(f"exact_match (token-based): {exact_match.item()}")

    if seq_is_correct.item() and not exact_match.item():
        print("\n✅ SUCCESS!")
        print("The fixed loss correctly recognizes the alternative optimal solution!")
        print("This will allow the model to converge by not penalizing optimal solutions.")
        return True
    else:
        print("\n❌ FAILED!")
        print("The fix didn't work as expected.")
        return False


def test_invalid_permutation():
    """Test that invalid permutations are correctly rejected."""

    print("\n" + "="*60)
    print("Testing Invalid Permutation Detection")
    print("="*60)

    matrix_size = 3
    cost_matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    # Valid assignment
    valid_assignment = np.array([0, 1, 2])
    # Invalid assignment (not a permutation - column 0 used twice)
    invalid_assignment = np.array([0, 0, 2])

    # Convert to tokens
    valid_tokens = create_lsa_tokens(valid_assignment, matrix_size)
    invalid_tokens = create_lsa_tokens(invalid_assignment, matrix_size)

    # Create inputs and labels
    inputs = torch.tensor(cost_matrix.flatten()).unsqueeze(0)
    labels = valid_tokens.unsqueeze(0)

    vocab_size = 102
    invalid_logits = create_fake_logits(invalid_tokens, vocab_size).unsqueeze(0)

    # Test the function
    seq_is_correct = cost_based_seq_is_correct(
        inputs=inputs,
        labels=labels,
        pred_logits=invalid_logits,
        matrix_size=matrix_size
    )

    print(f"\nValid assignment: {valid_assignment}")
    print(f"Invalid assignment: {invalid_assignment} (column 0 used twice)")
    print(f"seq_is_correct: {seq_is_correct.item()}")

    if not seq_is_correct.item():
        print("\n✅ SUCCESS!")
        print("Invalid permutations are correctly rejected.")
        return True
    else:
        print("\n❌ FAILED!")
        print("Invalid permutation was incorrectly accepted.")
        return False


def test_helper_functions():
    """Test the helper functions."""

    print("\n" + "="*60)
    print("Testing Helper Functions")
    print("="*60)

    matrix_size = 3
    assignment = np.array([2, 0, 1])  # row 0->col 2, row 1->col 0, row 2->col 1

    # Create tokens
    tokens = create_lsa_tokens(assignment, matrix_size)
    print(f"\nOriginal assignment: {assignment}")
    print(f"Tokens: {tokens.numpy()}")

    # Test conversion back
    tokens_batch = tokens.unsqueeze(0)
    recovered_assignment = binary_tokens_to_assignment(tokens_batch, matrix_size)
    print(f"Recovered assignment: {recovered_assignment[0].numpy()}")

    if np.array_equal(recovered_assignment[0].numpy(), assignment):
        print("✅ binary_tokens_to_assignment works correctly")
    else:
        print("❌ binary_tokens_to_assignment FAILED")

    # Test cost calculation
    cost_matrix = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=torch.float32).unsqueeze(0)

    cost = calculate_assignment_cost(cost_matrix, recovered_assignment)
    expected_cost = 3 + 4 + 8  # row 0->col 2 (3), row 1->col 0 (4), row 2->col 1 (8)
    print(f"\nCalculated cost: {cost.item()}")
    print(f"Expected cost: {expected_cost}")

    if abs(cost.item() - expected_cost) < 1e-6:
        print("✅ calculate_assignment_cost works correctly")
    else:
        print("❌ calculate_assignment_cost FAILED")

    # Test permutation validation
    valid = torch.tensor([[0, 1, 2], [2, 0, 1]])
    invalid = torch.tensor([[0, 0, 2], [1, 2, 3]])

    valid_result = is_valid_permutation(valid, matrix_size)
    invalid_result = is_valid_permutation(invalid, matrix_size)

    print(f"\nValid permutations: {valid.numpy()}")
    print(f"Is valid: {valid_result.numpy()}")
    print(f"Invalid permutations: {invalid.numpy()}")
    print(f"Is valid: {invalid_result.numpy()}")

    if valid_result.all() and not invalid_result.any():
        print("✅ is_valid_permutation works correctly")
    else:
        print("❌ is_valid_permutation FAILED")


if __name__ == "__main__":
    test_helper_functions()
    success1 = test_cost_based_correctness()
    success2 = test_invalid_permutation()

    print("\n" + "="*60)
    print("OVERALL RESULT:")
    print("="*60)
    if success1 and success2:
        print("✅ All tests passed!")
        print("\nThe fixed loss function correctly:")
        print("  1. Accepts alternative optimal solutions")
        print("  2. Rejects invalid permutations")
        print("  3. Aligns training with evaluation metrics")
    else:
        print("❌ Some tests failed")
