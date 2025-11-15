from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


# ===== LSA-SPECIFIC HELPER FUNCTIONS =====

def binary_tokens_to_assignment(tokens, matrix_size):
    """
    Convert binary token sequence to assignment vector.

    Args:
        tokens: (batch, seq_len) where seq_len = matrix_size^2, values are 1 or 2
        matrix_size: size of the assignment matrix

    Returns:
        assignment: (batch, matrix_size) where assignment[b,i] = j means row i -> column j
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
        assignments: (batch, matrix_size) where assignments[b,i] = j means row i -> column j

    Returns:
        costs: (batch,) total cost for each assignment
    """
    batch_size, matrix_size = assignments.shape
    # For each row i, get cost_matrix[i, assignment[i]]
    row_indices = torch.arange(matrix_size, device=assignments.device).unsqueeze(0).expand(batch_size, -1)
    batch_indices = torch.arange(batch_size, device=assignments.device).unsqueeze(1).expand(-1, matrix_size)

    costs_per_row = cost_matrices[batch_indices, row_indices, assignments]
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

    # Check if all values are in range [0, matrix_size)
    valid = (assignments >= 0).all(dim=1) & (assignments < matrix_size).all(dim=1)

    # Check if each column is used exactly once (permutation)
    # For each batch element, check if sorted assignment equals [0, 1, 2, ..., matrix_size-1]
    sorted_assignments = torch.sort(assignments, dim=1)[0]
    expected = torch.arange(matrix_size, device=assignments.device).unsqueeze(0).expand(batch_size, -1)
    valid = valid & (sorted_assignments == expected).all(dim=1)

    return valid


def cost_based_seq_is_correct(inputs, labels, pred_logits, matrix_size, ignore_label_id=-100):
    """
    Determine if a sequence is correct based on assignment cost (for LSA tasks).

    Instead of checking exact token matching, this checks if the predicted assignment
    achieves the same cost as the ground truth assignment.

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

    # Get ground truth assignment
    gt_assignment = binary_tokens_to_assignment(labels, matrix_size)

    # Get predicted tokens and convert to assignment
    pred_tokens = torch.argmax(pred_logits, dim=-1)
    pred_assignment = binary_tokens_to_assignment(pred_tokens, matrix_size)

    # Check if predictions are valid permutations
    valid_pred = is_valid_permutation(pred_assignment, matrix_size)

    # Calculate costs
    gt_cost = calculate_assignment_cost(cost_matrices, gt_assignment)

    # For invalid predictions, set cost to infinity
    pred_cost = torch.full((batch_size,), float('inf'), device=inputs.device, dtype=torch.float32)
    if valid_pred.any():
        pred_cost[valid_pred] = calculate_assignment_cost(
            cost_matrices[valid_pred],
            pred_assignment[valid_pred]
        )

    # Check if predicted cost equals ground truth cost (within tolerance)
    seq_is_correct = torch.abs(pred_cost - gt_cost) < 1e-4

    return seq_is_correct


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str, task_type: Optional[str] = None, matrix_size: int = 9):
        """
        Args:
            model: The underlying model
            loss_type: Type of loss function to use
            task_type: Type of task (e.g., 'lsa' for Linear Sum Assignment)
            matrix_size: Size of the assignment matrix for LSA tasks
        """
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.task_type = task_type
        self.matrix_size = matrix_size

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)

            # ===== MODIFIED: Use cost-based correctness for LSA =====
            if self.task_type == 'lsa' and "inputs" in new_carry.current_data:
                # Use cost-based correctness
                seq_is_correct = cost_based_seq_is_correct(
                    inputs=new_carry.current_data["inputs"],
                    labels=labels,
                    pred_logits=outputs["logits"],
                    matrix_size=self.matrix_size,
                    ignore_label_id=IGNORE_LABEL_ID
                )
            else:
                # Use exact token matching (original behavior)
                seq_is_correct = is_correct.sum(-1) == loss_counts
            # ===== END MODIFICATION =====

            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),

                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses

        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()
