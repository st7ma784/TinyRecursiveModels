# Linear Sum Assignment (LSA) Dataset

## Overview

This dataset builder creates Linear Sum Assignment problem instances following the same structure and augmentation strategy as the Sudoku dataset. The Linear Sum Assignment problem (also known as the Hungarian Algorithm problem) seeks to find an optimal assignment of workers to jobs (or rows to columns) that minimizes the total cost.

## Problem Description

Given an n×n cost matrix C where C[i,j] represents the cost of assigning row i to column j, find a permutation (assignment) that minimizes the total cost:

```
minimize: Σ C[i, assignment[i]] for i = 0..n-1
```

The solution is a permutation of {0, 1, ..., n-1} representing which column each row is assigned to.

## Dataset Structure

The dataset follows the same structure as the Sudoku dataset:

### Files Generated
- `train/dataset.json` - Metadata for training set
- `train/all__inputs.npy` - Training cost matrices (flattened to 1D)
- `train/all__labels.npy` - Training optimal assignments
- `train/all__puzzle_identifiers.npy` - Puzzle type identifiers
- `train/all__puzzle_indices.npy` - Indices for puzzle boundaries
- `train/all__group_indices.npy` - Indices for group boundaries
- `test/` - Same structure for test set
- `identifiers.json` - Mapping of identifier IDs to names

### Data Format

**Inputs**: Each cost matrix is a 9×9 matrix flattened to a sequence of 81 integers
- Shape: (num_samples, 81)
- Values: Integers from 1 to max_value (default 100)

**Labels**: Each optimal assignment is a permutation of length 9
- Shape: (num_samples, 9)  
- Values: Integers from 0 to 8 (representing column assignments)
- Constraint: Each label is a valid permutation

## Augmentation Strategy

Similar to `shuffle_sudoku`, the `shuffle_lsa` function creates augmented versions by:

1. **Row Permutation**: Randomly shuffle the rows of the cost matrix
2. **Column Permutation**: Randomly shuffle the columns of the cost matrix
3. **Assignment Update**: Update the optimal assignment to match the shuffled matrix

The key property: **The shuffled assignment remains optimal for the shuffled cost matrix with the same total cost**.

### How shuffle_lsa Works

```python
# Given original cost matrix C and assignment A
row_perm = random_permutation([0,1,...,n-1])
col_perm = random_permutation([0,1,...,n-1])

# Shuffle matrix: C'[i,j] = C[row_perm[i], col_perm[j]]
shuffled_cost = C[row_perm][:, col_perm]

# Update assignment accordingly
for each old_row i with assignment[i] = j:
    new_row = inverse_row_perm[i]
    new_col = inverse_col_perm[j]
    shuffled_assignment[new_row] = new_col
```

This ensures:
- The problem structure is preserved
- The assignment remains optimal
- The total cost is unchanged
- Augmented examples are mathematically equivalent but look different

## Usage

### Generate Dataset

```bash
# Basic usage with defaults (10k train, 1k test, 9×9 matrices)
python dataset/build_lsa_dataset.py preprocess_data

# Custom configuration
python dataset/build_lsa_dataset.py preprocess_data \
    --output_dir data/lsa-custom \
    --train_samples 50000 \
    --test_samples 5000 \
    --matrix_size 9 \
    --max_value 100 \
    --num_aug 5
```

### Configuration Parameters

- `output_dir`: Where to save the dataset (default: `data/lsa-9x9`)
- `train_samples`: Number of training samples (default: 10000)
- `test_samples`: Number of test samples (default: 1000)
- `matrix_size`: Size of the cost matrix (default: 9 for 9×9)
- `max_value`: Maximum value for cost matrix entries (default: 100)
- `num_aug`: Number of augmentations per training sample (default: 0)

### Load Dataset

The dataset can be loaded using the `PuzzleDataset` class:

```python
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig

config = PuzzleDatasetConfig(
    seed=42,
    dataset_paths=["data/lsa-9x9"],
    global_batch_size=32,
    test_set_mode=False,
    epochs_per_iter=1,
    rank=0,
    num_replicas=1
)

dataset = PuzzleDataset(config, split="train")
```

## Comparison with Sudoku Dataset

| Aspect | Sudoku | Linear Sum Assignment |
|--------|--------|----------------------|
| Input size | 81 (9×9 grid) | 81 (9×9 matrix) |
| Input values | 0-9 (digits) | 1-100 (costs) |
| Output size | 81 (9×9 solution) | 9 (assignment vector) |
| Output values | 1-9 (digits) | 0-8 (column indices) |
| Shuffle method | Digit mapping + transposition + band/stack permutation | Row/column permutation |
| Key property | Valid Sudoku → Valid Sudoku | Optimal assignment → Optimal assignment |

## Example Instance

**Original Cost Matrix** (5×5 for illustration):
```
[[14  3  3  7 18]
 [20 11  2  1 18]
 [16 10  1 15  1]
 [16 20 15  5  1]
 [17  5 18  4  3]]
```

**Optimal Assignment**: [0, 3, 2, 4, 1]
- Row 0 → Column 0 (cost 14)
- Row 1 → Column 3 (cost 1)
- Row 2 → Column 2 (cost 1)
- Row 3 → Column 4 (cost 1)
- Row 4 → Column 1 (cost 5)
- **Total Cost**: 22

**Shuffled Cost Matrix**:
```
[[ 2  1 18 11 20]
 [ 3  7 18  3 14]
 [ 1 15  1 10 16]
 [15  5  1 20 16]
 [18  4  3  5 17]]
```

**Shuffled Assignment**: [1, 4, 0, 2, 3]
- **Total Cost**: 22 (same!)

## Testing

Run the test script to verify the implementation:

```bash
python test_lsa_dataset.py
```

This will:
1. Test that shuffling preserves optimality
2. Verify assignments are valid permutations
3. Demonstrate the dataset with examples

## Integration with Training Pipeline

The LSA dataset is compatible with the existing training pipeline:
- Uses the same `PuzzleDatasetMetadata` structure
- Follows the same batching and sampling logic
- Supports the same augmentation framework
- Can be mixed with other puzzle datasets
