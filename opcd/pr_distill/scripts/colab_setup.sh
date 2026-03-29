#!/bin/bash
# Colab A100 80GB environment setup for PR-Distill experiments
# Uses uv for fast dependency management
set -x

DRIVE_DIR="/content/drive/MyDrive/pr_distill"
WORK_DIR="/content/LMOps"
DATA_DIR="/tmp/pr_distill_data"

echo "=== Step 1: Mount Google Drive ==="
python3 -c "from google.colab import drive; drive.mount('/content/drive')"

echo "=== Step 2: Install uv ==="
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "=== Step 3: Clone / Update codebase ==="
if [ -d "$WORK_DIR" ]; then
    cd $WORK_DIR && git pull
else
    git clone https://github.com/YTianZHU/LMOps.git $WORK_DIR
fi

echo "=== Step 4: Install verl + PR-Distill dependencies ==="
cd $WORK_DIR/opcd/verl && uv pip install --system -e ".[gpu,math]"
cd $WORK_DIR/opcd/pr_distill && uv pip install --system -e ".[train]"

echo "=== Step 5: Restore data from Google Drive ==="
mkdir -p $DATA_DIR
if [ -d "$DRIVE_DIR/data" ]; then
    cp -r $DRIVE_DIR/data/* $DATA_DIR/
    echo "Restored data from Google Drive"
else
    echo "No existing data found. Run prepare_data.sh first."
fi

echo "=== Step 6: Prepare DAPO-Math-17K (if not already present) ==="
if [ ! -f "$DATA_DIR/dapo_train_raw.parquet" ]; then
    echo "Downloading DAPO-Math-17K..."
    python3 -c "
from datasets import load_dataset
ds = load_dataset('ytz20/dapo_train')['train']
ds.to_parquet('$DATA_DIR/dapo_train_raw.parquet')
print(f'Train: {len(ds)} rows')
ds2 = load_dataset('ytz20/dapo_test')['train']
ds2.to_parquet('$DATA_DIR/dapo_test_raw.parquet')
print(f'Test: {len(ds2)} rows')
"
fi

echo "=== Step 7: Symlink data to /tmp ==="
if [ -f "$DATA_DIR/student_train.parquet" ]; then
    ln -sf $DATA_DIR/student_train.parquet /tmp/dapo_train.parquet
    ln -sf $DATA_DIR/test.parquet /tmp/dapo_test.parquet
fi

echo "=== Step 8: Login wandb ==="
if [ -n "$WANDB_API_KEY" ]; then
    wandb login $WANDB_API_KEY
else
    echo "Set WANDB_API_KEY to enable logging"
fi

echo "=== Setup complete ==="
echo "Data dir: $DATA_DIR"
echo "Work dir: $WORK_DIR"
nvidia-smi
