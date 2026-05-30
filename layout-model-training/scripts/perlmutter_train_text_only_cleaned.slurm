#!/bin/bash
#SBATCH -A atlas
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH -J yb_text_clean
#SBATCH -o logs/yb_text_clean_%j.out
#SBATCH -e logs/yb_text_clean_%j.err

set -euo pipefail

REPO_DIR="/global/cfs/projectdirs/atlas/haofeng/yearbook-urap-standalone"
VENV_DIR="${REPO_DIR}/.venv310"
PROJECT_ID="158111"

cd "${REPO_DIR}"
mkdir -p layout-model-training/logs layout-model-training/outputs/yearbook-text-only/fast_rcnn_R_50_FPN

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  echo "Missing virtual environment at ${VENV_DIR}"
  exit 1
fi

source "${VENV_DIR}/bin/activate"

# Avoid portalocker issues on shared filesystems when downloading Detectron2 model weights.
JOB_TMP_DIR="${SLURM_TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}"
mkdir -p "${JOB_TMP_DIR}/fvcore_cache"
export FVCORE_CACHE="${JOB_TMP_DIR}/fvcore_cache"

python -c "import torch, detectron2; print('Using', torch.__version__, detectron2.__version__)"

python code/labelstudio/import_export_sandbox/download_project_images.py \
  --project-id "${PROJECT_ID}" \
  --out-dir layout-model-training/data/yearbook/images

python create_clean_text_training_data.py \
  --annotations layout-model-training/data/yearbook/annotations.json \
  --images-dir layout-model-training/data/yearbook/images

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
