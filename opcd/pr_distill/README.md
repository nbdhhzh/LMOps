# Posterior-Refined Context Distillation (PR-Distill)

Experimental code for **Posterior-Refined Context Distillation**, built on top of [OPCD](../README.md).

Core idea: replace OPCD's accumulated experiential knowledge with **per-problem hints from a strong model (GLM-5)** as the teacher's context, grounded in Posterior Refinement theory (Theorem 2.2).

## Quick Start

### 1. Environment Setup (Colab A100 80GB)

```bash
bash opcd/pr_distill/scripts/colab_setup.sh
```

### 2. Data Preparation

```bash
# Download DAPO-Math-17K
bash opcd/pr_distill/scripts/prepare_data.sh

# Generate GLM-5 hints (requires API key)
python opcd/pr_distill/prepare_glm5_hints.py \
    --input /tmp/pr_distill_data/dapo_train_raw.parquet \
    --output /tmp/pr_distill_data/dapo_train_with_hints.parquet \
    --api_key YOUR_GLM5_API_KEY \
    --resume

# Build all dataset views
bash opcd/pr_distill/scripts/prepare_data.sh  # re-run to build views
```

### 3. Experiments

**SeqKD from GLM-5** (baseline):
```bash
bash opcd/pr_distill/scripts/pr_seqkd_1gpu.sh Qwen/Qwen3-1.7B
```

**PR-CD** (off-policy FKL):
```bash
# Step 1: Teacher generates with hints
bash opcd/pr_distill/scripts/pr_generate_offp_1gpu.sh Qwen/Qwen3-1.7B

# Step 2: Student trains on teacher responses
bash opcd/pr_distill/scripts/pr_train_offp_1gpu.sh Qwen/Qwen3-1.7B /tmp/pr-offp-gen-Qwen3-1.7B/off_policy_data
```

**PR-OPCD** (on-policy RKL, main method):
```bash
# Step 1: Extract
bash opcd/pr_distill/scripts/pr_extract_1gpu.sh Qwen/Qwen3-1.7B

# Step 2: Consolidate
bash opcd/pr_distill/scripts/pr_consolidate_1gpu.sh Qwen/Qwen3-1.7B /path/to/experience
```

## Experiments Overview

| Method | Context Source | KL Type | Script |
|--------|---------------|---------|--------|
| SeqKD from GLM-5 | GLM-5 response (SFT) | CE | `pr_seqkd_1gpu.sh` |
| PR-CD (off-policy) | GLM-5 hint → teacher | full (RKL) | `pr_generate_offp` + `pr_train_offp` |
| PR-OPCD (on-policy) | GLM-5 hint → teacher | full (RKL) | `pr_extract` + `pr_consolidate` |

## Key Differences from OPCD

| | OPCD | PR-Distill |
|---|------|-----------|
| Context c | Accumulated experiential knowledge | Per-problem GLM-5 solution |
| Context source | Self-extracted (iterative) | External strong model (one-shot) |
| Theoretical grounding | Empirical | Theorem 2.2 (posterior >= prior) |
| Training framework | Same | Same |

## File Structure

```
pr_distill/
├── README.md
├── prepare_glm5_hints.py     # GLM-5 API calls
├── build_pr_dataset.py        # Dataset construction
└── scripts/
    ├── colab_setup.sh         # Colab environment
    ├── prepare_data.sh        # Data download + build
    ├── pr_seqkd_1gpu.sh      # SeqKD baseline
    ├── pr_generate_offp_1gpu.sh  # PR-CD: teacher generation
    ├── pr_train_offp_1gpu.sh    # PR-CD: student training
    ├── pr_extract_1gpu.sh       # PR-OPCD: extract
    └── pr_consolidate_1gpu.sh   # PR-OPCD: consolidate
```

## Code Changes to verl

Minimal modification to `opcd/verl/verl/trainer/ppo/ray_trainer.py`:
- Added `PR_HINT_SOLVE_PROMPT_TEMPLATE` for hint injection
- Added `use_per_sample_hint` config option in consolidate loop
- When enabled, reads per-sample `glm5_hint` from data instead of global experience

All original OPCD functionality is preserved (backward compatible).
