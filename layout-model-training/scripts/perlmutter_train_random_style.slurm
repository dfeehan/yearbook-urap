#!/bin/bash
#SBATCH -A atlas
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH -J yb_randstyle
#SBATCH -o logs/random-style/%x_%j.out
#SBATCH -e logs/random-style/%x_%j.err

set -euo pipefail

REPO_DIR="/global/cfs/projectdirs/atlas/haofeng/yearbook-urap-standalone"
VENV_DIR="${REPO_DIR}/.venv310"

# Required env vars (injected by submit_random_style_runs.sh)
RUN_ID="${RUN_ID:?RUN_ID is required}"
TRAIN_JSON="${TRAIN_JSON:?TRAIN_JSON is required}"
VAL_JSON="${VAL_JSON:-data/yearbook/annotations-text-only-clean-val.json}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
SAMPLE_SIZE="${SAMPLE_SIZE:-unknown}"
SEED="${SEED:-unknown}"

IMAGES_DIR="data/yearbook/images"
CONFIG="configs/yearbook/fast_rcnn_R_50_FPN_task_ids.yaml"

cd "${REPO_DIR}"
mkdir -p layout-model-training/logs/random-style
mkdir -p "layout-model-training/${OUTPUT_DIR}"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  echo "Missing virtual environment at ${VENV_DIR}"
  exit 1
fi

source "${VENV_DIR}/bin/activate"

JOB_TMP_DIR="${SLURM_TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}"
mkdir -p "${JOB_TMP_DIR}/fvcore_cache"
export FVCORE_CACHE="${JOB_TMP_DIR}/fvcore_cache"

if [ ! -f "layout-model-training/${TRAIN_JSON}" ]; then
  echo "Missing train JSON: layout-model-training/${TRAIN_JSON}"
  exit 1
fi
if [ ! -f "layout-model-training/${VAL_JSON}" ]; then
  echo "Missing val JSON: layout-model-training/${VAL_JSON}"
  exit 1
fi

echo "======================================"
echo "Random-style run: ${RUN_ID}"
echo "Sample size     : ${SAMPLE_SIZE}"
echo "Seed            : ${SEED}"
echo "Train JSON      : ${TRAIN_JSON}"
echo "Val JSON        : ${VAL_JSON}"
echo "Output dir      : ${OUTPUT_DIR}"
echo "======================================"
python -c "import torch, detectron2; print('torch', torch.__version__, '| detectron2', detectron2.__version__, '| CUDA', torch.cuda.is_available())"

echo "Train: $(python -c "import json; d=json.load(open('layout-model-training/${TRAIN_JSON}')); print(len(d['images']), 'images,', len(d['annotations']), 'boxes')")"
echo "Val  : $(python -c "import json; d=json.load(open('layout-model-training/${VAL_JSON}')); print(len(d['images']), 'images,', len(d['annotations']), 'boxes')")"

cd "${REPO_DIR}/layout-model-training"

python tools/train_net.py \
  --config-file "${CONFIG}" \
  --num-gpus 1 \
  --dataset_name random_style_${RUN_ID} \
  --json_annotation_train "${TRAIN_JSON}" \
  --image_path_train "${IMAGES_DIR}" \
  --json_annotation_val "${VAL_JSON}" \
  --image_path_val "${IMAGES_DIR}" \
  OUTPUT_DIR "${OUTPUT_DIR}"
