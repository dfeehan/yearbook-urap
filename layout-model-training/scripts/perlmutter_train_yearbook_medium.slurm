#!/bin/bash
#SBATCH -A atlas
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH -J yb_medium_train
#SBATCH -o logs/yb_medium_%j.out
#SBATCH -e logs/yb_medium_%j.err

set -euo pipefail

# ---- YEARBOOK-ONLY PATHS (keep separate from your HEP project) ----
REPO_DIR="/global/cfs/projectdirs/atlas/haofeng/yearbook-urap-standalone"
CONDA_ENV="/global/cfs/projectdirs/atlas/haofeng/conda_envs/yearbook_env"

cd "${REPO_DIR}/layout-model-training"
mkdir -p logs outputs/yearbook-medium-2class/fast_rcnn_R_50_FPN_3x

# If your conda is initialized differently, replace with your own init command.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python tools/train_net.py \
  --num-gpus 1 \
  --dataset_name yearbook-layout-medium \
  --json_annotation_train data/yearbook/annotations-train-medium.json \
  --image_path_train data/yearbook/images \
  --json_annotation_val data/yearbook/annotations-val-medium.json \
  --image_path_val data/yearbook/images \
  --config-file configs/yearbook/fast_rcnn_R_50_FPN_3x.yaml \
  OUTPUT_DIR outputs/yearbook-medium-2class/fast_rcnn_R_50_FPN_3x/ \
  SOLVER.IMS_PER_BATCH 2
