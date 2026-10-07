#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4           # Request 4 CPU cores for dataloading
#SBATCH --mem=32G                   # Request 32GB total memory
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --time=04:00:00             # Max time limit (inference is fast, 4 hours is plenty)
#SBATCH --job-name=camlc_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# 1. Setup the environment (Uncomment the one you are using)
source ~/miniconda/bin/activate brset-camlc

# 2. Make sure the logs directory exists
mkdir -p logs

# 3. Setup Seed and Isolation
# Pass the seed as the first argument to the script (defaults to 42 if not provided)
SEED=${1:-42}
RUNS_DIR="/data/BRSET/runs/7L-${SEED}"

echo "Starting evaluation for runs in: $RUNS_DIR"

# 4. Run the evaluation script
python evaluate_all.py \
    --root "/data/BRSET" \
    --runs-dir "$RUNS_DIR" \
    --test-manifest "image_evaluation_12.csv" \
    --image-size 512 \
    --device "cuda"

echo "Evaluation complete! Results saved to $RUNS_DIR/test_evaluation_results.json"
