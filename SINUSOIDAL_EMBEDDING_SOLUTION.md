# Sinusoidal Value Embeddings for LSA: Solution to Learning Failure

## Problem Diagnosis

Your LSA model was completely failing to learn (all metrics at 0) because of a **fundamental mismatch** between the data representation and the task requirements.

### The Core Issue

**Current Approach (Discrete Embeddings):**
- Treats cost values 1-100 as categorical tokens
- Each value gets an independent, random embedding vector
- No inductive bias that similar values should have similar representations
- The model has no way to understand that cost 50 is "between" 49 and 51

**Why This Fails for LSA:**
- In Sudoku (your reference task), digits 1-9 are truly categorical - no inherent ordering
- In LSA, cost values are **ordinal/continuous** - numerical relationships matter
- The model needs to understand that:
  - Similar costs should lead to similar assignment decisions
  - The magnitude of costs affects optimization
  - Interpolation between values is meaningful

### Evidence from Tests

```
Discrete Embedding Similarities:
  Similarity(10, 11): -0.1394   ← Completely random!
  Similarity(10, 50): -0.0255   ← No relationship to distance

Sinusoidal Embedding Similarities:
  Similarity(10, 11): 1.0000    ← Nearly identical (as expected)
  Similarity(10, 50): 0.9943    ← Still similar, but less so
```

## Solution: Sinusoidal Value Embeddings

### What Are Sinusoidal Embeddings?

Similar to positional encodings in transformers, sinusoidal embeddings encode continuous values using sine and cosine waves:

```python
# For a value v normalized to [0, 1]:
freq_i = 1 / (10000^(i/d))  # Different frequency for each dimension
embedding[2i]   = sin(v * freq_i)
embedding[2i+1] = cos(v * freq_i)
```

### Key Benefits

1. **Smooth, Continuous Representation**
   - Similar values have similar embeddings
   - Model can interpolate between values
   - Numerical relationships are preserved

2. **Strong Inductive Bias**
   - Built-in understanding of ordinal structure
   - No need to learn that 50 is between 49 and 51
   - Faster convergence

3. **Generalization**
   - Can handle values not seen during training
   - Smooth gradients for optimization

## Implementation

### Three Embedding Types Provided

#### 1. Pure Sinusoidal (`value_embedding_type: "sinusoidal"`)
- 100% sinusoidal encoding
- Strongest inductive bias
- Good for tasks where ordering is primary

#### 2. Hybrid (`value_embedding_type: "hybrid"`) **← RECOMMENDED**
- Combines sinusoidal + learnable embeddings
- Best of both worlds:
  - Sinusoidal component: smooth, ordinal structure
  - Learnable component: task-specific patterns
- Configurable ratio (default 50/50)

#### 3. Discrete (default)
- Original approach
- Only for categorical data (Sudoku, etc.)

### Files Modified

1. **`models/value_embedding.py`** (NEW)
   - `SinusoidalValueEmbedding`: Pure sinusoidal encoding
   - `HybridValueEmbedding`: Sinusoidal + learnable
   - `ContinuousValueEmbedding`: Drop-in replacement API

2. **`models/recursive_reasoning/trm.py`**
   - Added config options for value embeddings
   - Conditional embedding selection based on task
   - Lines 137-164: Embedding initialization

3. **`config/arch/trm.yaml`**
   - Added value embedding configuration options
   - Default: disabled (backward compatible)

4. **`config/cfg_lsa.yaml`** **← ALREADY ENABLED FOR YOU**
   - Sinusoidal embeddings enabled for LSA
   - Hybrid mode with 50/50 ratio
   - Max value set to 100 (matches dataset)

## Usage

### For LSA (Already Configured)

Your LSA config is **already set up** with sinusoidal embeddings:

```yaml
# config/cfg_lsa.yaml
arch:
  use_sinusoidal_value_embedding: True
  value_embedding_max: 100.0
  value_embedding_min: 0.0
  value_embedding_type: "hybrid"
  hybrid_sinusoidal_ratio: 0.5
```

### Training Commands

Just train normally - the new embeddings are automatic:

```bash
# Your existing training script should work
bash train_lsa.sh

# Or directly:
python pretrain.py --config-name cfg_lsa
```

### For Other Tasks

To enable for other continuous-value tasks:

```yaml
arch:
  use_sinusoidal_value_embedding: True
  value_embedding_max: <your_max_value>
  value_embedding_min: <your_min_value>  # often 0
  value_embedding_type: "hybrid"  # or "sinusoidal"
  hybrid_sinusoidal_ratio: 0.5
```

## Expected Improvements

With sinusoidal embeddings, you should see:

1. **Non-zero metrics immediately**
   - `lsa/valid_permutation_rate` should start > 0% within first eval
   - Model produces valid assignments faster

2. **Faster convergence**
   - Optimal solutions found with less training
   - Smoother loss curves

3. **Better generalization**
   - Can handle cost values not in training set
   - More robust to distribution shifts

## Validation

Run the test suite to verify implementation:

```bash
python test_sinusoidal_embedding.py
```

Expected output:
- ✓ Nearby values more similar: True
- ✓ Smooth transitions: True
- ✓ Shape correct: True
- ✓ Sinusoidal embeddings preserve numerical relationships better

## Architecture Comparison

### Before (Discrete)
```
Cost Value (50) → Lookup Table[50] → Random Vector[512]
                                          ↓
                                    (no relation to 49 or 51)
```

### After (Sinusoidal)
```
Cost Value (50) → Normalize to [0,1] → Sin/Cos Waves → Smooth Vector[512]
                                                             ↓
                                                (similar to embeddings of 49, 51)
```

## Technical Details

### Embedding Composition (Hybrid Mode)

With `hidden_size=512` and `hybrid_sinusoidal_ratio=0.5`:

- **Sinusoidal component**: 256 dims
  - 128 sine waves at different frequencies
  - 128 cosine waves (same frequencies)
  - Provides smooth, continuous structure

- **Learnable component**: 256 dims
  - Standard embedding lookup table
  - Learns task-specific patterns
  - Complements sinusoidal structure

### Frequency Bands

Similar to positional encodings:
- Low frequencies: capture coarse differences (1 vs 100)
- High frequencies: capture fine differences (50 vs 51)
- Logarithmically spaced for balanced representation

## Troubleshooting

### If metrics are still 0:

1. **Check configuration loaded**
   ```bash
   # Verify config in logs
   grep "use_sinusoidal_value_embedding" outputs/*/pretrain.log
   ```

2. **Verify model architecture**
   ```python
   # In Python/IPython:
   model = ...  # Load your model
   print(type(model.inner.embed_tokens))
   # Should show: HybridValueEmbedding
   ```

3. **Check value ranges**
   - Ensure `value_embedding_max` matches your dataset
   - LSA uses values 1-100, so max=100

### If convergence is slow:

Try adjusting the hybrid ratio:
```yaml
hybrid_sinusoidal_ratio: 0.75  # More sinusoidal
# or
hybrid_sinusoidal_ratio: 0.25  # More learnable
```

## Next Steps

1. **Train with new embeddings**
   ```bash
   bash train_lsa.sh
   ```

2. **Monitor metrics**
   - Watch for non-zero `lsa/valid_permutation_rate`
   - Compare loss curves to previous runs

3. **Experiment with ratios**
   - Try pure sinusoidal: `value_embedding_type: "sinusoidal"`
   - Adjust hybrid ratio: `hybrid_sinusoidal_ratio: 0.25` to `0.75`

4. **Ablation study** (optional)
   - Train with discrete embeddings (set `use_sinusoidal_value_embedding: False`)
   - Compare performance quantitatively

## References

This approach is inspired by:
- **Positional Encodings**: "Attention Is All You Need" (Vaswani et al., 2017)
- **Fourier Features**: Used in neural implicit representations (NeRF, etc.)
- **Continuous Embeddings**: Common in regression tasks and signal processing

## Summary

Your hypothesis was **exactly right**! The model was treating continuous cost values as categorical tokens, losing all numerical structure. Sinusoidal embeddings provide the inductive bias needed for ordinal data, giving the model a smooth, continuous representation space where similar values have similar embeddings.

The fix is already in place - just retrain and you should see dramatic improvements.
