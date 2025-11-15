"""
Option 1: Cost-Based Correctness for LSA

Modify the loss computation to use cost-based correctness instead of exact token matching.
This requires computing the assignment cost during training.
"""

import torch
import torch.nn.functional as F


def binary_tokens_to_assignment(tokens, matrix_size):
    """
    Convert binary token sequence to assignment vector.

    Args:
        tokens: (batch, seq_len) where seq_len = matrix_size^2, values are 1 or 2
        matrix_size: size of the assignment matrix

    Returns:
        assignment: (batch, matrix_size) where assignment[i,j] means row i -> column j
    """
    batch_size = tokens.shape[0]
    # Subtract 1 to get 0-1 binary matrix
    binary_matrix = (tokens - 1).reshape(batch_size, matrix_size, matrix_size)
    # For each row, find column with value 1 (argmax)
    assignment = torch.argmax(binary_matrix, dim=-1)
    return assignment


def calculate_assignment_cost(cost_matrices, assignments):
    """
    Calculate the cost of assignments.

    Args:
        cost_matrices: (batch, matrix_size, matrix_size)
        assignments: (batch, matrix_size) where assignments[i,j] means row i -> column j

    Returns:
        costs: (batch,) total cost for each assignment
    """
    batch_size, matrix_size = assignments.shape
    # Gather costs for each assignment
    # For each row i, get cost_matrix[i, assignment[i]]
    row_indices = torch.arange(matrix_size, device=assignments.device).unsqueeze(0).expand(batch_size, -1)
    costs_per_row = cost_matrices[torch.arange(batch_size, device=assignments.device).unsqueeze(1),
                                   row_indices,
                                   assignments]
    return costs_per_row.sum(dim=-1)


def is_valid_permutation(assignments, matrix_size):
    """
    Check if assignments are valid permutations.

    Args:
        assignments: (batch, matrix_size)
        matrix_size: size of the assignment

    Returns:
        valid: (batch,) boolean mask indicating valid permutations
    """
    batch_size = assignments.shape[0]
    valid = torch.ones(batch_size, dtype=torch.bool, device=assignments.device)

    # Check if all values are in range [0, matrix_size)
    valid &= (assignments >= 0).all(dim=1)
    valid &= (assignments < matrix_size).all(dim=1)

    # Check if each column is used exactly once (permutation)
    for b in range(batch_size):
        if valid[b]:
            unique_vals = torch.unique(assignments[b])
            valid[b] = (len(unique_vals) == matrix_size)

    return valid


def cost_based_seq_is_correct(inputs, labels, pred_logits, matrix_size, ignore_label_id=-100):
    """
    Determine if a sequence is correct based on assignment cost.

    Args:
        inputs: (batch, seq_len) cost matrices flattened
        labels: (batch, seq_len) ground truth assignment as binary tokens (1 or 2)
        pred_logits: (batch, seq_len, vocab_size) model predictions
        matrix_size: size of the assignment matrix
        ignore_label_id: label id to ignore

    Returns:
        seq_is_correct: (batch,) boolean indicating if prediction is optimal
    """
    batch_size = inputs.shape[0]

    # Get cost matrices
    cost_matrices = inputs.reshape(batch_size, matrix_size, matrix_size).float()

    # Get ground truth and predicted assignments
    gt_assignment = binary_tokens_to_assignment(labels, matrix_size)

    # Get predictions
    pred_tokens = torch.argmax(pred_logits, dim=-1)
    pred_assignment = binary_tokens_to_assignment(pred_tokens, matrix_size)

    # Check if predictions are valid permutations
    valid_pred = is_valid_permutation(pred_assignment, matrix_size)

    # Calculate costs
    gt_cost = calculate_assignment_cost(cost_matrices, gt_assignment)

    # For invalid predictions, set cost to infinity
    pred_cost = torch.full((batch_size,), float('inf'), device=inputs.device)
    if valid_pred.any():
        pred_cost[valid_pred] = calculate_assignment_cost(
            cost_matrices[valid_pred],
            pred_assignment[valid_pred]
        )

    # Check if predicted cost equals ground truth cost (within tolerance)
    seq_is_correct = torch.abs(pred_cost - gt_cost) < 1e-4

    return seq_is_correct


# Example of how to integrate this into ACTLossHead
def modified_forward_example(inputs, labels, logits, matrix_size=9, ignore_label_id=-100):
    """
    Example showing how to modify the forward pass in ACTLossHead.
    """
    # Original code calculates this:
    # is_correct = (torch.argmax(logits, dim=-1) == labels)
    # seq_is_correct = is_correct.sum(-1) == loss_counts

    # New code should calculate this instead:
    seq_is_correct = cost_based_seq_is_correct(
        inputs, labels, logits, matrix_size, ignore_label_id
    )

    return seq_is_correct


print("Option 1: Cost-Based Correctness")
print("="*60)
print("This approach modifies seq_is_correct to check if the predicted")
print("assignment achieves the same cost as the ground truth, rather")
print("than checking for exact token matching.")
print()
print("Pros:")
print("  + Simple conceptual change")
print("  + Aligns training with evaluation")
print("  + Model rewarded for finding ANY optimal solution")
print()
print("Cons:")
print("  - Requires computing costs during training (extra computation)")
print("  - Need to reshape tokens back to assignment matrix")
print("  - Need to handle invalid permutations gracefully")
