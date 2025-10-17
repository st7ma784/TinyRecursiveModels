# LSA Training Status Report

## Summary

Successfully set up Linear Sum Assignment (LSA) problem for training with the Tiny Recursive Models framework. The dataset has been generated and the training infrastructure is configured, though we encountered a tensor reshaping issue during model execution.

## What Was Accomplished

### 1. Dataset Creation ✓
- Created `dataset/build_lsa_dataset.py` following Sudoku dataset structure
- Implemented `shuffle_lsa()` function for data augmentation via row/column permutations
- Generated dataset with 10,000 training samples + 10 augmentations each = 110,000 total samples
- Generated 1,000 test samples
- Dataset location: `data/lsa-9x9-10k/`

### 2. Data Format ✓
- **Inputs**: 9×9 cost matrices (flattened to 81 values)
- **Labels**: Binary assignment matrices (flattened to 81 values)
  - Value 1 = unassigned position
  - Value 2 = assigned position
- Both inputs and labels have shape (N, 81) for compatibility with the model

### 3. Evaluator Implementation ✓
- Created `evaluators/lsa.py` with LSA-specific metrics:
  - Exact match accuracy
  - Valid permutation rate
  - Optimal cost accuracy
  - Average cost difference
- Evaluator properly decodes binary matrices back to assignment vectors

### 4. Configuration ✓
- Created `config/cfg_lsa.yaml` adapted from Sudoku config
- Training parameters:
  - Batch size: 128
  - Learning rate: 1e-4
  - Epochs: 50,000
  - Eval interval: 5,000
  - EMA enabled with rate 0.999

### 5. Training Script ✓
- Created `train_lsa.sh` for easy launching
- Configured to run on GPU 0
- Uses screen session named "LSARun"

### 6. Dependencies ✓
- Installed: argdantic, hydra-core, omegaconf, wandb, coolname, einops
- Modified pretrain.py to fallback to torch.optim.Adam (adam_atan2 not available)

## Current Issue

### Tensor Reshaping Error
The training encounters a runtime error in the attention mechanism:

```
Cannot view a tensor with shape torch.Size([128, 97, 8, 64]) and strides (49664, 64, 6208, 1) 
as a tensor with shape (128, 97, 512)!
```

**Root Cause**: The tensor has non-contiguous memory layout (strides don't match expected pattern) due to torch dynamo compilation

**Location**: `/data/TinyRecursiveModels/models/layers.py:134`
```python
attn_output = attn_output.view(batch_size, seq_len, self.output_size)
```

## Solutions to Try

### Option 1: Make Tensor Contiguous (Recommended)
Modify `models/layers.py` line 134:
```python
attn_output = attn_output.contiguous().view(batch_size, seq_len, self.output_size)
```

### Option 2: Disable Torch Compile
Add to training script or config:
```python
torch._dynamo.config.suppress_errors = True
# or
export TORCHDYNAMO_DISABLE=1
```

### Option 3: Adjust Sequence Length
The issue might be related to seq_len=81 not being ideal for the model. Consider:
- Padding to 96 (divisible by more factors)
- Checking if model expects specific sequence lengths

## How to Resume

### Quick Fix and Restart:
```bash
# Kill current training
pkill -f "python pretrain.py.*lsa"

# Apply fix to models/layers.py (add .contiguous())

# Restart training
cd /data/TinyRecursiveModels
screen -RR LSARun
# or
bash train_lsa.sh
```

### Check Training Progress:
```bash
tail -f /data/TinyRecursiveModels/lsa_training.log
ps aux | grep pretrain
screen -r LSARun  # Attach to screen session
```

## Files Created

1. `/data/TinyRecursiveModels/dataset/build_lsa_dataset.py` - Dataset builder
2. `/data/TinyRecursiveModels/dataset/README_LSA.md` - Documentation
3. `/data/TinyRecursiveModels/evaluators/lsa.py` - Evaluator
4. `/data/TinyRecursiveModels/config/cfg_lsa.yaml` - Training config
5. `/data/TinyRecursiveModels/train_lsa.sh` - Training script
6. `/data/TinyRecursiveModels/test_lsa_dataset.py` - Test suite
7. `/data/TinyRecursiveModels/visualize_lsa_dataset.py` - Visualization tool
8. `/data/TinyRecursiveModels/compare_shuffle_logic.py` - Comparison doc
9. `/data/TinyRecursiveModels/LSA_IMPLEMENTATION_SUMMARY.md` - Implementation summary

## Dataset Statistics

- Training samples: 110,000 (10,000 × 11)
- Test samples: 1,000
- Input/Label shape: (N, 81)
- Vocab size: 102 (0=PAD + 1-100 for costs + special)
- Matrix size: 9×9
- Max cost value: 100

## Next Steps

1. **Fix tensor reshaping issue** (add `.contiguous()`)
2. **Restart training** and monitor initial metrics
3. **Evaluate first checkpoint** at 5,000 steps
4. **Analyze results**:
   - Check if model learns valid permutations
   - Monitor optimal cost accuracy
   - Compare with random baseline

## Expected Training Time

Based on Sudoku example (< 36 hours on single L40S):
- Estimated: ~24-48 hours on single GPU
- Can monitor via wandb dashboard
- Checkpoints saved every 5,000 steps
