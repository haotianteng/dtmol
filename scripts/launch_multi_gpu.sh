#!/bin/bash
# Launch multi-GPU training on 8×H100 via torchrun (PyTorch elastic launch).
#
# Usage:
#   # Full 8-GPU run on NaCl multi-grid dataset
#   bash scripts/launch_multi_gpu.sh --head-mode linear --steps 50000
#
#   # 4-GPU subset
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NGPU=4 bash scripts/launch_multi_gpu.sh
#
#   # Unit-cell dataset (real crystals)
#   bash scripts/launch_multi_gpu.sh --dataset-mode unified \
#       --datamix dtmol/data/datamix_unit_cell.yaml \
#       --datamix-val dtmol/data/datamix_unit_cell_val.yaml
#
# Environment variables:
#   NGPU          Number of GPUs (default: auto-detect)
#   MASTER_PORT   DDP port (default: 29500)
#   EXTRA_ARGS    Additional args passed to the training script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "=============================================="
echo " dtmol multi-GPU training"
echo " GPUs:         $NGPU"
echo " Master port:  $MASTER_PORT"
echo " Extra args:   $*"
echo "=============================================="

# Default training args — override via CLI
# With 8 GPUs × batch_size=1 × gradient_accumulation=4:
#   effective batch = 8 × 1 × 4 = 32 per optimizer step
DEFAULT_ARGS=(
    --dataset-mode unified
    --datamix dtmol/data/datamix_unit_cell.yaml
    --datamix-val dtmol/data/datamix_unit_cell_val.yaml
    --device cuda
    --batch-size 1
    --epochs 100
    --lr 1e-3
    --lambda-force 0
    --lambda-fd-force 0
    --dropout 0.0
    --gradient-accumulation-steps 4
    --head-mode linear
    --use-wandb
)

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun \
    --nproc_per_node="$NGPU" \
    --master_port="$MASTER_PORT" \
    dtmol/dtmol_train_test.py \
    "${DEFAULT_ARGS[@]}" \
    "$@"
