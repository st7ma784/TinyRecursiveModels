# LSA Training Bug Fix - Summary

## The Bug: Training-Evaluation Mismatch

### Root Cause

The model was **not converging** because of a fundamental mismatch between the training loss and evaluation metrics:

**Training Loss (`models/losses.py:194-208`):**
- Used **exact token-by-token matching** to determine if a sequence is correct
- `seq_is_correct = (predicted_tokens == ground_truth_tokens).all()`

**Evaluation Metric (`evaluators/lsa.py:112`):**
- Used **cost-based comparison** to determine if a solution is optimal
- `is_optimal = (predicted_cost == ground_truth_cost)`

### The Problem

Linear Sum Assignment problems can have **multiple optimal solutions** with the same minimal cost. Even when the model finds an alternative optimal assignment, it would be:

1. ✅ **Marked as CORRECT** by the evaluator (same cost)
2. ❌ **Marked as INCORRECT** by the training loss (different tokens)

This caused the `q_halt_loss` to be trained on the **wrong target**, preventing convergence.

### Demonstration

```python
# Cost matrix with multiple optimal solutions
cost_matrix = [[1, 2, 3],
               [1, 2, 3],  # Duplicate row allows swapping
               [3, 2, 1]]

# Ground truth: [0, 1, 2] with cost = 4
# Alternative:  [1, 0, 2] with cost = 4  (equally optimal!)

# But the tokens are different:
# GT tokens:  [2, 1, 1, 1, 2, 1, 1, 1, 2]
# Alt tokens: [1, 2, 1, 2, 1, 1, 1, 1, 2]
#              ^  ^     ^  ^              <- 4 differences

# Training would mark this as INCORRECT (0% accuracy)
# Evaluation would mark this as CORRECT (100% accuracy)
```

## The Fix

### Modified Files

1. **`models/losses.py`** - Added cost-based correctness checking for LSA
2. **`config/arch/trm.yaml`** - Added task_type and matrix_size parameters
3. **`config/cfg_lsa.yaml`** - Enabled LSA mode

### Key Changes

#### 1. Added Helper Functions (`models/losses.py:41-153`)

```python
def cost_based_seq_is_correct(inputs, labels, pred_logits, matrix_size):
    """
    Checks if predicted assignment achieves optimal cost,
    rather than checking exact token match.
    """
    # Convert tokens to assignments
    gt_assignment = binary_tokens_to_assignment(labels, matrix_size)
    pred_assignment = binary_tokens_to_assignment(pred_tokens, matrix_size)

    # Calculate costs
    gt_cost = calculate_assignment_cost(cost_matrices, gt_assignment)
    pred_cost = calculate_assignment_cost(cost_matrices, pred_assignment)

    # Check if optimal (within tolerance)
    return abs(pred_cost - gt_cost) < 1e-4
```

#### 2. Modified ACTLossHead (`models/losses.py:157-209`)

```python
class ACTLossHead(nn.Module):
    def __init__(self, model, loss_type, task_type=None, matrix_size=9):
        # Added task_type and matrix_size parameters

    def forward(self, ...):
        # Modified to use cost-based correctness for LSA:
        if self.task_type == 'lsa' and "inputs" in new_carry.current_data:
            seq_is_correct = cost_based_seq_is_correct(...)
        else:
            seq_is_correct = is_correct.sum(-1) == loss_counts
```

#### 3. Updated Configuration

**`config/cfg_lsa.yaml`:**
```yaml
arch:
  loss:
    task_type: lsa  # Enable cost-based correctness
    matrix_size: 9  # 9x9 assignment matrix
```

## Benefits

✅ **Aligns training with evaluation** - Both use cost-based correctness
✅ **Accepts alternative optimal solutions** - Model not penalized for finding different but equally good solutions
✅ **Backward compatible** - Other tasks (Sudoku, etc.) use original exact matching
✅ **Properly validates permutations** - Invalid assignments are correctly rejected

## Results

**Before Fix:**
- Model finds optimal solution: `[1, 0, 2]` (cost = 4)
- seq_is_correct = False (token mismatch)
- q_halt_loss penalizes the model
- Model doesn't learn when to halt

**After Fix:**
- Model finds optimal solution: `[1, 0, 2]` (cost = 4)
- seq_is_correct = True (cost matches GT)
- q_halt_loss rewards the model
- Model learns to halt on optimal solutions

## Your Question: Temperature Sampling Approach

You asked about using **multiple predictions with temperature** and analyzing each - this is actually an **excellent idea**! It's related to:

### Reward-Based Training (Option 2)

Instead of binary "correct/incorrect", you could:

1. **Sample multiple solutions** from the model using temperature
2. **Evaluate each solution's cost**
3. **Use cost as a reward signal** to train the model

This is similar to **policy gradient methods** (REINFORCE, PPO):

```python
# Sample multiple solutions
solutions = model.sample(temperature=1.0, num_samples=10)

# Evaluate each
rewards = [calculate_reward(sol) for sol in solutions]

# Train to increase probability of better solutions
loss = -log_prob * (reward - baseline)
```

### Advantages
- Directly optimizes for solution quality (cost)
- Naturally handles multiple optimal solutions
- Could explore the solution space better

### Implementation
See `proposed_fix_option2.py` for a detailed implementation sketch.

## How to Use the Fix

1. **For LSA tasks:** The fix is already configured in `config/cfg_lsa.yaml`
2. **For other tasks:** Leave `task_type: null` (uses original exact matching)
3. **To train:** Run your normal training command - the fix is automatic

## Testing

Run the test suite to verify the fix:
```bash
python test_fixed_loss.py
```

Expected output:
```
✅ All tests passed!

The fixed loss function correctly:
  1. Accepts alternative optimal solutions
  2. Rejects invalid permutations
  3. Aligns training with evaluation metrics
```

## Next Steps

1. **Retrain the model** with the fixed loss function
2. **Monitor metrics:**
   - `exact_accuracy` - How often exact token match (may be lower)
   - `lsa/optimal_cost_acc` - How often optimal cost found (should improve!)
   - `q_halt_accuracy` - Should improve significantly
3. **Consider Option 2** if you want to explore reward-based training further

---

## Summary

The model wasn't converging because it was being trained to match exact tokens but evaluated on cost optimality. The fix aligns these by using cost-based correctness in training for LSA tasks, allowing the model to be rewarded for finding ANY optimal solution, not just the specific ground truth sequence.
