#!/bin/bash
#SBATCH -A atlas
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH -J yb_text_clean
#SBATCH -o logs/yb_text_clean_%j.out
#SBATCH -e logs/yb_text_clean_%j.err

set -euo pipefail

REPO_DIR="/global/cfs/projectdirs/atlas/haofeng/yearbook-urap-standalone"
CONDA_ENV="/global/cfs/projectdirs/atlas/haofeng/conda_envs/yearbook_env"
PROJECT_ID="158111"

cd "${REPO_DIR}"
mkdir -p layout-model-training/logs layout-model-training/outputs/yearbook-text-only/fast_rcnn_R_50_FPN

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python code/labelstudio/import_export_sandbox/download_project_images.py \
  --project-id "${PROJECT_ID}" \
  --out-dir layout-model-training/data/yearbook/images

python create_clean_text_training_data.py

cd "${REPO_DIR}/layout-model-training"
python tools/train_net.py \
  --config-file configs/yearbook/fast_rcnn_R_50_FPN_text_only.yaml \
  --num-gpus 1 \
  --dataset_name yearbook_text_only_clean \
  --json_annotation_train data/yearbook/annotations-text-only-clean-train.json \
  --image_path_train data/yearbook/images \
  --json_annotation_val data/yearbook/annotations-text-only-clean-val.json \
  --image_path_val data/yearbook/images \
  OUTPUT_DIR outputs/yearbook-text-only/fast_rcnn_R_50_FPN/
