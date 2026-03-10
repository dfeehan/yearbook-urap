#!/bin/bash
#SBATCH -A atlas
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH -J yb_gpu_check
#SBATCH -o logs/gpu_check_%j.out
#SBATCH -e logs/gpu_check_%j.err

set -euo pipefail

# ---- YEARBOOK-ONLY PATHS (keep separate from your HEP project) ----
REPO_DIR="/global/cfs/projectdirs/atlas/haofeng/yearbook-urap-standalone"
CONDA_ENV="/global/cfs/projectdirs/atlas/haofeng/conda_envs/yearbook_env"

mkdir -p "${REPO_DIR}/layout-model-training/logs"

# If your conda is initialized differently, replace with your own init command.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python - << 'PY'
import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu name:", torch.cuda.get_device_name(0))
PY
