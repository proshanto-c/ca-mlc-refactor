#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4           # Request 4 CPU cores for dataloading
#SBATCH --mem=32G                   # Request 32GB total memory
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --time=47:59:00             # Max time limit
#SBATCH --job-name=camlc_tuning
#SBATCH --output=logs/tune_%A_%a.out
#SBATCH --error=logs/tune_%A_%a.err

# 1. Setup the environment (Uncomment the one you are using)
# If using Miniconda:
source ~/miniconda/bin/activate brset-camlc

# If using Python Venv:
# source ~/ca-mlc-refactor/brset-camlc/bin/activate

# 2. Make sure the logs directory exists
mkdir -p logs

# 3. Setup Seed and Isolation
# Pass the seed as the first argument to the script (defaults to 42 if not provided)
SEED=${1:-42}
PROJECT="ResNet-GNN"
OUTPUT_DIR="/users/sann7128/ca-mlc-refactor/data/BRSET/runs/tuning_seed${SEED}"

# 4. Run the tuning script
# Each job gets its own unique SLURM_ARRAY_TASK_ID (0 to 26)
python tune_all.py \
    --root "/users/sann7128/ca-mlc-refactor/data/BRSET" \
    --output-dir "$OUTPUT_DIR" \
    --seed $SEED \
    --wandb-project "$PROJECT" \
    --split-total 27 \
    --split-index $SLURM_ARRAY_TASK_ID \
    --num-trials 1
