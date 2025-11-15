"""
Linear Sum Assignment Evaluator

Evaluates the model's predictions on LSA problems by:
1. Checking if predicted assignments are valid permutations
2. Verifying if they are optimal (match the ground truth cost)
3. Computing accuracy metrics
"""

from typing import Dict, Optional
import torch
import numpy as np
import torch.distributed as dist

from dataset.common import PuzzleDatasetMetadata


class LSA:
    """Evaluator for Linear Sum Assignment problems."""
    
    required_outputs = {"inputs", "labels", "preds"}
    
    def __init__(self,
                 data_path: str,
                 eval_metadata: PuzzleDatasetMetadata,
                 matrix_size: int = 9):
        super().__init__()
        self.matrix_size = matrix_size
        self.seq_len = matrix_size * matrix_size
        self.blank_identifier_id = eval_metadata.blank_identifier_id
        
        # Metrics storage
        self._local_metrics = {
            'total_samples': 0,
            'exact_match': 0,
            'valid_permutation': 0,
            'optimal_cost': 0,
            'cost_diff_sum': 0.0,
        }
    
    def begin_eval(self):
        """Reset metrics at the start of evaluation."""
        self._local_metrics = {
            'total_samples': 0,
            'exact_match': 0,
            'valid_permutation': 0,
            'optimal_cost': 0,
            'cost_diff_sum': 0.0,
        }
    
    def _binary_matrix_to_assignment(self, binary_matrix: np.ndarray) -> np.ndarray:
        """Convert binary assignment matrix back to assignment vector."""
        # binary_matrix is flattened, reshape it
        matrix = binary_matrix.reshape(self.matrix_size, self.matrix_size)
        # Subtract 1 to get back to 0-1 values
        matrix = matrix - 1
        # For each row, find which column has value 1
        assignment = np.argmax(matrix, axis=1)
        return assignment
    
    def _is_valid_permutation(self, assignment: np.ndarray) -> bool:
        """Check if assignment is a valid permutation."""
        if len(assignment) != self.matrix_size:
            return False
        # Check if it contains exactly the values 0..n-1
        return len(np.unique(assignment)) == self.matrix_size and \
               np.all((assignment >= 0) & (assignment < self.matrix_size))
    
    def _calculate_cost(self, cost_matrix: np.ndarray, assignment: np.ndarray) -> float:
        """Calculate the total cost of an assignment."""
        if not self._is_valid_permutation(assignment):
            return float('inf')
        return float(sum(cost_matrix[i, assignment[i]] for i in range(self.matrix_size)))
    
    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        """Update metrics with a batch of predictions."""
        # Move to CPU and convert to numpy
        inputs = batch["inputs"].cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        pred_labels = preds["preds"].cpu().numpy()

        # Filter out padding if using puzzle_identifiers
        # Note: Only filter if there are actual padded samples (not all blank_identifier_id)
        if "puzzle_identifiers" in batch:
            puzzle_ids = batch["puzzle_identifiers"].cpu().numpy()
            mask = puzzle_ids != self.blank_identifier_id

            # Only apply filter if there's at least one valid sample
            # (This handles the case where blank_identifier_id == 0 and all real puzzles also have id 0)
            if mask.any():
                inputs = inputs[mask]
                labels = labels[mask]
                pred_labels = pred_labels[mask]

        batch_size = len(inputs)
        self._local_metrics['total_samples'] += batch_size
        
        for i in range(batch_size):
            # Reshape input to matrix
            cost_matrix = inputs[i].reshape(self.matrix_size, self.matrix_size)
            
            # Convert binary matrices to assignment vectors
            true_assignment = self._binary_matrix_to_assignment(labels[i])
            pred_assignment = self._binary_matrix_to_assignment(pred_labels[i])
            
            # Check exact match
            if np.array_equal(true_assignment, pred_assignment):
                self._local_metrics['exact_match'] += 1
            
            # Check if valid permutation
            if self._is_valid_permutation(pred_assignment):
                self._local_metrics['valid_permutation'] += 1
                
                # Calculate costs
                true_cost = self._calculate_cost(cost_matrix, true_assignment)
                pred_cost = self._calculate_cost(cost_matrix, pred_assignment)
                
                # Check if optimal (same cost as ground truth)
                if abs(pred_cost - true_cost) < 1e-6:
                    self._local_metrics['optimal_cost'] += 1
                
                # Track cost difference
                self._local_metrics['cost_diff_sum'] += abs(pred_cost - true_cost)
    
    def result(self, save_path: Optional[str], rank: int, world_size: int,
               group: Optional[torch.distributed.ProcessGroup] = None) -> Optional[Dict[str, float]]:
        """Aggregate results across all processes and compute final metrics."""

        # Gather metrics from all ranks
        if world_size > 1:
            all_metrics = [None for _ in range(world_size)] if rank == 0 else None
            dist.gather_object(self._local_metrics, all_metrics, dst=0, group=group)
        else:
            # Single GPU mode - no need to gather
            all_metrics = [self._local_metrics]

        # Only rank 0 computes and returns final metrics
        if rank != 0:
            return None

        # Aggregate metrics
        aggregated = {
            'total_samples': 0,
            'exact_match': 0,
            'valid_permutation': 0,
            'optimal_cost': 0,
            'cost_diff_sum': 0.0,
        }

        for metrics in all_metrics:  # type: ignore
            for key in aggregated.keys():
                aggregated[key] += metrics[key]
        
        # Compute percentages and averages
        total = max(aggregated['total_samples'], 1)  # Avoid division by zero
        
        results = {
            'lsa/total_samples': float(aggregated['total_samples']),
            'lsa/exact_match_acc': 100.0 * aggregated['exact_match'] / total,
            'lsa/valid_permutation_rate': 100.0 * aggregated['valid_permutation'] / total,
            'lsa/optimal_cost_acc': 100.0 * aggregated['optimal_cost'] / total,
            'lsa/avg_cost_diff': aggregated['cost_diff_sum'] / total,
        }
        
        # Print results
        print("\n" + "="*60)
        print("Linear Sum Assignment Evaluation Results")
        print("="*60)
        print(f"Total Samples: {aggregated['total_samples']}")
        print(f"Exact Match Accuracy: {results['lsa/exact_match_acc']:.2f}%")
        print(f"Valid Permutation Rate: {results['lsa/valid_permutation_rate']:.2f}%")
        print(f"Optimal Cost Accuracy: {results['lsa/optimal_cost_acc']:.2f}%")
        print(f"Average Cost Difference: {results['lsa/avg_cost_diff']:.4f}")
        print("="*60 + "\n")
        
        return results
