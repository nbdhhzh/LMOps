#!/bin/bash
# Download DAPO-Math-17K and prepare all PR-Distill datasets
# Prerequisites: Run prepare_glm5_hints.py first to generate GLM-5 hints
set -x

DATA_DIR="${DATA_DIR:-/tmp/pr_distill_data}"
WORK_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

mkdir -p $DATA_DIR

echo "=== Step 1: Download DAPO-Math-17K ==="
if [ ! -f "$DATA_DIR/dapo_train_raw.parquet" ]; then
    cd $WORK_DIR
    python3 -c "
from datasets import load_dataset
ds_train = load_dataset('ytz20/dapo_train')['train']
ds_train.to_parquet('$DATA_DIR/dapo_train_raw.parquet')
print(f'Saved train: {len(ds_train)} rows')

ds_test = load_dataset('ytz20/dapo_test')['train']
ds_test.to_parquet('$DATA_DIR/dapo_test_raw.parquet')
print(f'Saved test: {len(ds_test)} rows')
"
else
    echo "DAPO data already exists, skipping download"
fi

echo "=== Step 2: Generate GLM-5 hints ==="
if [ ! -f "$DATA_DIR/dapo_train_with_hints.parquet" ]; then
    echo "Run the following command to generate hints:"
    echo "  python3 $WORK_DIR/opcd/pr_distill/prepare_glm5_hints.py \\"
    echo "      --input $DATA_DIR/dapo_train_raw.parquet \\"
    echo "      --output $DATA_DIR/dapo_train_with_hints.parquet \\"
    echo "      --api_key YOUR_API_KEY \\"
    echo "      --resume"
    echo ""
    echo "After hints are generated, re-run this script."
    exit 0
else
    echo "Hints already generated"
fi

echo "=== Step 3: Build PR-Distill datasets ==="
python3 $WORK_DIR/opcd/pr_distill/build_pr_dataset.py \
    --train_input $DATA_DIR/dapo_train_with_hints.parquet \
    --test_input $DATA_DIR/dapo_test_raw.parquet \
    --output_dir $DATA_DIR

echo "=== Step 4: Symlink for verl compatibility ==="
ln -sf $DATA_DIR/student_train.parquet /tmp/dapo_train.parquet
ln -sf $DATA_DIR/test.parquet /tmp/dapo_test.parquet

echo "=== Data preparation complete ==="
ls -lh $DATA_DIR/*.parquet
