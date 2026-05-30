#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$TRAINING_ROOT/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv310"
PYTHON_BIN="$VENV_DIR/bin/python"

PROJECT_ID=253267
TRAIN_JSON="data/yearbook/annotations-clean60-train.json"
VAL_JSON="data/yearbook/annotations-clean60-val.json"
IMAGES_DIR="data/yearbook/images"
OVERLAP_REPORT="data/yearbook/annotations-clean60-overlap-review.csv"
CONFIG="configs/yearbook/fast_rcnn_R_50_FPN_clean60.yaml"
OUTPUT_DIR="outputs/yearbook-clean60/fast_rcnn_R_50_FPN"
NUM_GPUS=0

if [ ! -x "$PYTHON_BIN" ]; then
  echo "❌ Missing Python environment at $PYTHON_BIN"
  exit 1
fi

cd "$PROJECT_ROOT"
source "$VENV_DIR/bin/activate"

if python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
  NUM_GPUS=1
fi

# Avoid portalocker issues when Detectron2 downloads pretrained weights.
JOB_TMP_DIR="${TMPDIR:-/tmp/${USER}/yearbook_clean60}"
mkdir -p "$JOB_TMP_DIR/fvcore_cache"
export FVCORE_CACHE="$JOB_TMP_DIR/fvcore_cache"

echo "======================================"
echo "Clean 60-Image Text Training"
echo "======================================"
python -c "import torch, detectron2; print('torch', torch.__version__, '| detectron2', detectron2.__version__, '| CUDA', torch.cuda.is_available())"
echo ""
echo "Regenerating train/val split from Label Studio project ${PROJECT_ID}..."

python create_project_text_dataset.py \
  --project-id "$PROJECT_ID"

echo ""
echo "Train: $(python -c "import json; d=json.load(open('layout-model-training/${TRAIN_JSON}')); print(len(d['images']), 'images,', len(d['annotations']), 'boxes')")"
echo "Val  : $(python -c "import json; d=json.load(open('layout-model-training/${VAL_JSON}')); print(len(d['images']), 'images,', len(d['annotations']), 'boxes')")"
echo "Overlap review: layout-model-training/${OVERLAP_REPORT}"
echo ""
echo "Config    : ${CONFIG}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Max iters : 2500 (~104 effective epochs on 47 train images)"
echo "Base LR   : 0.00025"
echo "GPUs      : ${NUM_GPUS}"
echo ""

cd "$TRAINING_ROOT"
mkdir -p logs "$OUTPUT_DIR"

python tools/train_net.py \
  --config-file "$CONFIG" \
  --num-gpus "$NUM_GPUS" \
  --dataset_name yearbook_clean60 \
  --json_annotation_train "$TRAIN_JSON" \
  --image_path_train "$IMAGES_DIR" \
  --json_annotation_val "$VAL_JSON" \
  --image_path_val "$IMAGES_DIR" \
  OUTPUT_DIR "$OUTPUT_DIR"

python tools/train_net.py \
  --config-file "$CONFIG" \
  --eval-only \
  --num-gpus "$NUM_GPUS" \
  --dataset_name yearbook_clean60_eval \
  --json_annotation_train "$TRAIN_JSON" \
  --image_path_train "$IMAGES_DIR" \
  --json_annotation_val "$VAL_JSON" \
  --image_path_val "$IMAGES_DIR" \
  MODEL.WEIGHTS "$OUTPUT_DIR/model_final.pth" \
  OUTPUT_DIR "$OUTPUT_DIR"