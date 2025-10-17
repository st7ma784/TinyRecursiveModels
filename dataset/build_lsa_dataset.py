from typing import Optional
import os
import json
import numpy as np
from scipy.optimize import linear_sum_assignment

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

from common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    output_dir: str = "data/lsa-9x9"
    
    # Dataset generation parameters
    train_samples: int = 10000
    test_samples: int = 1000
    matrix_size: int = 9
    
    # Value generation parameters
    max_value: int = 100  # Maximum value for cost matrix entries
    
    # Augmentation
    num_aug: int = 0


def shuffle_lsa(cost_matrix: np.ndarray, assignment: np.ndarray):
    """
    Shuffle a Linear Sum Assignment instance similar to shuffle_sudoku.
    
    This function applies random permutations to rows and columns of the cost matrix
    and updates the assignment accordingly.
    
    Args:
        cost_matrix: (N, N) array representing the cost matrix
        assignment: (N,) array where assignment[i] = j means row i is assigned to column j
    
    Returns:
        Tuple of (shuffled_cost_matrix, shuffled_assignment)
    """
    n = cost_matrix.shape[0]
    
    # Generate random row and column permutations
    row_perm = np.random.permutation(n)
    col_perm = np.random.permutation(n)
    
    # Create inverse permutations for assignment update
    inv_row_perm = np.argsort(row_perm)
    inv_col_perm = np.argsort(col_perm)
    
    # Shuffle the cost matrix
    # New matrix[i, j] = Old matrix[row_perm[i], col_perm[j]]
    shuffled_cost = cost_matrix[row_perm][:, col_perm]
    
    # Update assignment
    # If old assignment: row i -> col j
    # New assignment: row inv_row_perm[i] -> col inv_col_perm[j]
    # We need to express this as new_assignment[new_row] = new_col
    # For each old row i with assignment j:
    #   new row position = inv_row_perm[i]
    #   new col position = inv_col_perm[j]
    shuffled_assignment = np.zeros(n, dtype=np.int32)
    for old_row in range(n):
        old_col = assignment[old_row]
        new_row = inv_row_perm[old_row]
        new_col = inv_col_perm[old_col]
        shuffled_assignment[new_row] = new_col
    
    return shuffled_cost, shuffled_assignment


def generate_lsa_instance(n: int, max_value: int):
    """
    Generate a random Linear Sum Assignment instance.
    
    Args:
        n: Matrix size (n x n)
        max_value: Maximum value for cost matrix entries
    
    Returns:
        Tuple of (cost_matrix, optimal_assignment)
    """
    # Generate random cost matrix
    cost_matrix = np.random.randint(1, max_value + 1, size=(n, n))
    
    # Solve using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Convert to assignment array format
    assignment = np.zeros(n, dtype=np.int32)
    assignment[row_ind] = col_ind
    
    return cost_matrix, assignment


def convert_subset(set_name: str, num_samples: int, config: DataProcessConfig):
    """
    Generate and save a dataset subset for Linear Sum Assignment.
    """
    inputs = []
    labels = []
    
    print(f"Generating {num_samples} {set_name} samples...")
    for _ in tqdm(range(num_samples)):
        cost_matrix, assignment = generate_lsa_instance(config.matrix_size, config.max_value)
        inputs.append(cost_matrix)
        labels.append(assignment)
    
    # Generate dataset with augmentations
    num_augments = config.num_aug if set_name == "train" else 0

    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0
    
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)
    
    for orig_inp, orig_out in zip(tqdm(inputs, desc=f"Processing {set_name}"), labels):
        for aug_idx in range(1 + num_augments):
            # First index is not augmented
            if aug_idx == 0:
                inp, out = orig_inp, orig_out
            else:
                inp, out = shuffle_lsa(orig_inp, orig_out)

            # Push puzzle (only single example)
            results["inputs"].append(inp)
            results["labels"].append(out)
            example_id += 1
            puzzle_id += 1
            
            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)
            
        # Push group
        results["group_indices"].append(puzzle_id)
    
    # Convert to numpy arrays
    # For inputs (cost matrices): flatten each n×n matrix to length n²
    seq_len = config.matrix_size * config.matrix_size
    inputs_array = np.stack([arr.flatten() for arr in results["inputs"]], axis=0)
    
    # Verify values are in valid range
    assert np.all((inputs_array >= 1) & (inputs_array <= config.max_value))
    
    # For labels: Convert assignment vector to n×n binary matrix
    # assignment[i] = j means row i is assigned to column j
    # So we create a matrix where position (i,j) has value 1 if assigned, 0 otherwise
    # Then add 1 to make it 1-indexed (0 for unassigned, which won't occur)
    labels_list = []
    for assignment in results["labels"]:
        # Verify assignment is valid permutation
        assert len(np.unique(assignment)) == config.matrix_size, f"Invalid assignment"
        # Create binary assignment matrix
        binary_matrix = np.zeros((config.matrix_size, config.matrix_size), dtype=np.int32)
        for i, j in enumerate(assignment):
            binary_matrix[i, j] = 1
        # Flatten and add 1 to make it 1-indexed (so values are 1 or 2)
        labels_list.append(binary_matrix.flatten() + 1)
    
    labels_array = np.stack(labels_list, axis=0)
    
    results = {
        "inputs": inputs_array.astype(np.int32),
        "labels": labels_array.astype(np.int32),
        
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata
    # vocab_size: 0 (PAD) + 1 (unassigned/0 in binary) + 2 (assigned/1 in binary)
    # Since we use binary matrix (0 or 1) + 1, we have values 1 or 2
    metadata = PuzzleDatasetMetadata(
        seq_len=seq_len,  # n×n for both input matrix and output assignment matrix
        vocab_size=config.max_value + 2,  # PAD + values 1..max_value for inputs, but labels use 1-2
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"]
    )

    # Save metadata as JSON
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)
        
    # Save data
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    
    print(f"Saved {set_name} dataset to {save_dir}")
    print(f"  Input shape: {results['inputs'].shape}")
    print(f"  Label shape: {results['labels'].shape}")
        
    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    """Generate Linear Sum Assignment dataset."""
    convert_subset("train", config.train_samples, config)
    convert_subset("test", config.test_samples, config)


if __name__ == "__main__":
    cli()
