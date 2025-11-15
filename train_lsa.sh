#!/bin/bash
# Training script for Linear Sum Assignment dataset
# This script launches the training on GPU 0

export CUDA_VISIBLE_DEVICES=0,1,2
export DISABLE_COMPILE=1  # Disable torch.compile for older GPU (P100)

run_name="pretrain_lsa_9x9_trm"

echo "Starting LSA training on GPU 0"
echo "Run name: $run_name"
echo "Dataset: data/lsa-9x9-10k"
echo "Note: Torch compilation disabled for GPU compatibility (P100)"
echo ""

python pretrain.py \
  --config-name cfg_lsa \
  arch=trm \
  data_paths="[data/lsa-9x9-10k]" \
  evaluators="[{name: lsa@LSA}]" \
  epochs=500 \
  eval_interval=100 \
  global_batch_size=128 \
  lr=1e-4 \
  puzzle_emb_lr=1e-4 \
  weight_decay=0.01 \
  puzzle_emb_weight_decay=0.01 \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=6 \
  arch.halt_exploration_prob=0.3 \
  +run_name=${run_name} \
  ema=True
