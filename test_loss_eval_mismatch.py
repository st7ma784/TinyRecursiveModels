"""
Test to demonstrate the training-evaluation mismatch in LSA.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def binary_matrix_to_tokens(assignment, matrix_size):
    """Convert assignment vector to token sequence (as done in dataset generation)."""
    binary_matrix = np.zeros((matrix_size, matrix_size), dtype=np.int32)
    for i, j in enumerate(assignment):
        binary_matrix[i, j] = 1
    # Flatten and add 1 to make it 1-indexed (values are 1 or 2)
    return binary_matrix.flatten() + 1


def calculate_cost(cost_matrix, assignment):
    """Calculate total cost of an assignment."""
    return sum(cost_matrix[i, assignment[i]] for i in range(len(assignment)))


def test_mismatch():
    """
    Test case showing training-eval mismatch.
    Even if a model predicts an optimal solution, it could be marked as incorrect
    in training if it doesn't match the exact token sequence.
    """
    print("="*60)
    print("Testing Training-Evaluation Mismatch")
    print("="*60)

    # Use the example from earlier with multiple optimal solutions
    cost_matrix = np.array([
        [1, 2, 3],
        [1, 2, 3],  # Same as row 0
        [3, 2, 1]
    ])

    # Ground truth assignment (what scipy gives us)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    gt_assignment = np.zeros(3, dtype=int)
    gt_assignment[row_ind] = col_ind

    # Alternative optimal assignment (swap rows 0 and 1)
    alt_assignment = np.array([gt_assignment[1], gt_assignment[0], gt_assignment[2]])

    # Convert to token sequences
    gt_tokens = binary_matrix_to_tokens(gt_assignment, 3)
    alt_tokens = binary_matrix_to_tokens(alt_assignment, 3)

    # Calculate costs
    gt_cost = calculate_cost(cost_matrix, gt_assignment)
    alt_cost = calculate_cost(cost_matrix, alt_assignment)

    print(f"\nCost matrix:")
    print(cost_matrix)
    print(f"\nGround truth assignment: {gt_assignment} (cost: {gt_cost})")
    print(f"Ground truth tokens: {gt_tokens}")
    print(f"\nAlternative optimal assignment: {alt_assignment} (cost: {alt_cost})")
    print(f"Alternative tokens: {alt_tokens}")

    # Check what happens in training vs evaluation
    print(f"\n{'='*60}")
    print("Training Loss (exact token match):")
    print(f"{'='*60}")

    exact_match = np.array_equal(gt_tokens, alt_tokens)
    print(f"Do tokens match exactly? {exact_match}")
    print(f"Would training loss consider this CORRECT? {exact_match}")

    print(f"\n{'='*60}")
    print("Evaluation Metric (optimal cost):")
    print(f"{'='*60}")

    is_optimal = abs(gt_cost - alt_cost) < 1e-6
    print(f"Is alternative solution optimal (same cost)? {is_optimal}")
    print(f"Would evaluation consider this CORRECT? {is_optimal}")

    print(f"\n{'='*60}")
    print("CONCLUSION:")
    print(f"{'='*60}")

    if is_optimal and not exact_match:
        print("⚠️  MISMATCH DETECTED!")
        print("The model could predict an OPTIMAL solution but still be")
        print("penalized in training because it doesn't match exact tokens!")
        print("\nThis explains why the model isn't converging:")
        print("- The q_halt_loss uses exact match as the target")
        print("- But the model should halt when it finds ANY optimal solution")
        print("- Not just when it matches the specific GT tokens")
        return True
    else:
        print("No mismatch in this case.")
        return False


def test_current_loss_function():
    """
    Simulate what the current loss function does with alternative optimal solutions.
    """
    print("\n" + "="*60)
    print("Simulating Current Loss Function Behavior")
    print("="*60)

    cost_matrix = np.array([
        [1, 2, 3],
        [1, 2, 3],
        [3, 2, 1]
    ])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    gt_assignment = np.zeros(3, dtype=int)
    gt_assignment[row_ind] = col_ind
    alt_assignment = np.array([gt_assignment[1], gt_assignment[0], gt_assignment[2]])

    gt_tokens = torch.tensor(binary_matrix_to_tokens(gt_assignment, 3))
    alt_tokens = torch.tensor(binary_matrix_to_tokens(alt_assignment, 3))

    # Simulate the current loss computation
    # Assume model predicted alternative optimal solution
    predicted_tokens = alt_tokens

    # This is what happens in losses.py:70-71
    is_correct = (predicted_tokens == gt_tokens)
    seq_is_correct = is_correct.sum() == len(gt_tokens)

    print(f"Predicted tokens: {predicted_tokens.numpy()}")
    print(f"Ground truth tokens: {gt_tokens.numpy()}")
    print(f"Token-wise match: {is_correct.numpy()}")
    print(f"Sequence is correct (for q_halt_loss target): {seq_is_correct.item()}")

    # Calculate actual costs
    pred_cost = calculate_cost(cost_matrix, alt_assignment)
    gt_cost = calculate_cost(cost_matrix, gt_assignment)

    print(f"\nPredicted assignment cost: {pred_cost}")
    print(f"Ground truth assignment cost: {gt_cost}")
    print(f"Is prediction optimal? {pred_cost == gt_cost}")

    print(f"\n⚠️  The model found an OPTIMAL solution (cost {pred_cost})")
    print(f"But seq_is_correct = {seq_is_correct.item()}, so it will be penalized!")
    print(f"The q_halt_loss will be trained on the WRONG target.")


if __name__ == "__main__":
    has_mismatch = test_mismatch()
    if has_mismatch:
        test_current_loss_function()
