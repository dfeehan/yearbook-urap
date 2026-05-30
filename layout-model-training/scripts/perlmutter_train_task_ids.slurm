#!/bin/bash
#SBATCH -A atlas
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH -J yb_task_ids
#SBATCH -o logs/yb_task_ids_%j.out
#SBATCH -e logs/yb_task_ids_%j.err

# ── Training run: 19 ayantee1939negr pages, Name + Additional Text only ───────
#    Train : layout-model-training/data/yearbook/annotations-task-ids-text.json
#    Val   : layout-model-training/data/yearbook/annotations-task-ids-val.json
#    Config: configs/yearbook/fast_rcnn_R_50_FPN_task_ids.yaml
#    Output: outputs/yearbook-task-ids/fast_rcnn_R_50_FPN/
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_DIR="/global/cfs/projectdirs/atlas/haofeng/yearbook-urap-standalone"
VENV_DIR="${REPO_DIR}/.venv310"
TRAIN_JSON="data/yearbook/annotations-task-ids-text.json"
VAL_JSON="data/yearbook/annotations-task-ids-val.json"
IMAGES_DIR="data/yearbook/images"
CONFIG="configs/yearbook/fast_rcnn_R_50_FPN_task_ids.yaml"
OUTPUT_DIR="outputs/yearbook-task-ids/fast_rcnn_R_50_FPN"

cd "${REPO_DIR}"
mkdir -p layout-model-training/logs layout-model-training/"${OUTPUT_DIR}"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  echo "❌ Missing virtual environment at ${VENV_DIR}"
  exit 1
fi

source "${VENV_DIR}/bin/activate"

# Avoid portalocker issues on Lustre when downloading Detectron2 model weights.
JOB_TMP_DIR="${SLURM_TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}"
mkdir -p "${JOB_TMP_DIR}/fvcore_cache"
export FVCORE_CACHE="${JOB_TMP_DIR}/fvcore_cache"

echo "======================================"
echo "Task-ID Text Training"
echo "======================================"
python -c "import torch, detectron2; print('torch', torch.__version__, '| detectron2', detectron2.__version__, '| CUDA', torch.cuda.is_available())"

# Verify annotation and image files exist
for f in "layout-model-training/${TRAIN_JSON}" "layout-model-training/${VAL_JSON}"; do
  if [ ! -f "${REPO_DIR}/${f}" ]; then
    echo "❌ Missing file: ${f}"
    echo "   Run: python3 create_task_id_text_dataset.py <task_ids> --download-images"
    exit 1
  fi
done

echo "Train: $(python -c "import json; d=json.load(open('layout-model-training/${TRAIN_JSON}')); print(len(d['images']), 'images,', len(d['annotations']), 'boxes')")"
echo "Val  : $(python -c "import json; d=json.load(open('layout-model-training/${VAL_JSON}')); print(len(d['images']), 'images,', len(d['annotations']), 'boxes')")"
echo ""
echo "Config    : ${CONFIG}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Max iters : 2000  (~200 effective epochs on 19 images)"
echo "Base LR   : 0.00025"
echo ""

cd "${REPO_DIR}/layout-model-training"

python tools/train_net.py \
  --config-file "${CONFIG}" \
  --num-gpus 1 \
  --dataset_name yearbook_task_ids \
  --json_annotation_train "${TRAIN_JSON}" \
  --image_path_train      "${IMAGES_DIR}" \
  --json_annotation_val   "${VAL_JSON}" \
  --image_path_val        "${IMAGES_DIR}" \
  OUTPUT_DIR "${OUTPUT_DIR}"
