#!/bin/bash
#SBATCH -A atlas
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH -J yb_clean60
#SBATCH -o logs/yb_clean60_%j.out
#SBATCH -e logs/yb_clean60_%j.err

set -euo pipefail

REPO_DIR="/global/cfs/projectdirs/atlas/haofeng/yearbook-urap-standalone"

cd "$REPO_DIR"
mkdir -p layout-model-training/logs

bash layout-model-training/scripts/train_clean60.sh