#!/bin/bash

# Train text-only model after filtering suspicious giant/portrait-like text boxes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$TRAINING_ROOT/.." && pwd)"
ANNOTATIONS_PATH="$TRAINING_ROOT/data/yearbook/annotations.json"
IMAGES_PATH="$TRAINING_ROOT/data/yearbook/images"
PYTHON_BIN="python3"
NUM_GPUS=0

if [ -x "$PROJECT_ROOT/.venv310/bin/python" ]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv310/bin/python"
elif [ -x "$PROJECT_ROOT/.venv311/bin/python" ]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv311/bin/python"
fi

cd "$TRAINING_ROOT"

if "$PYTHON_BIN" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
  NUM_GPUS=1
fi

echo "======================================"
echo "Training Cleaned Text-Only Model"
echo "======================================"
echo ""
echo "Dataset: text-only after suspicious-box filtering"
echo "  Train images: 329 | annotations: 12,146"
echo "  Val images: 83 | annotations: 3,189"
echo "Iterations: 7,000 (~21 iterations per train image)"
echo "GPUs: $NUM_GPUS"
echo "Expected training time: ~1-2 hours on 1 GPU, longer on CPU"
echo ""

if ! "$PYTHON_BIN" -c "import detectron2" >/dev/null 2>&1; then
  echo "❌ detectron2 is not installed in this Python environment."
  echo "   Expected interpreter: $PYTHON_BIN"
  echo "   Install dependencies in that environment, then retry training."
  exit 1
fi

if [ ! -f "$ANNOTATIONS_PATH" ]; then
  echo "❌ Missing annotations file: $ANNOTATIONS_PATH"
  echo "   Run: python3 $PROJECT_ROOT/fetch_annotations_direct.py"
  exit 1
fi

if [ ! -d "$IMAGES_PATH" ]; then
  echo "❌ Missing images directory: $IMAGES_PATH"
  echo "   Download project images before training."
  exit 1
fi

"$PYTHON_BIN" "$PROJECT_ROOT/create_clean_text_training_data.py" \
  --annotations "$ANNOTATIONS_PATH" \
  --images-dir "$IMAGES_PATH"

"$PYTHON_BIN" tools/train_net.py \
  --config-file "$TRAINING_ROOT/configs/yearbook/fast_rcnn_R_50_FPN_text_only.yaml" \
  --num-gpus "$NUM_GPUS" \
  --dataset_name yearbook_text_only_clean \
  --json_annotation_train "$TRAINING_ROOT/data/yearbook/annotations-text-only-clean-train.json" \
  --image_path_train "$IMAGES_PATH" \
  --json_annotation_val "$TRAINING_ROOT/data/yearbook/annotations-text-only-clean-val.json" \
  --image_path_val "$IMAGES_PATH" \
  2>&1 | tee "$TRAINING_ROOT/scripts/train_text_only_cleaned.log"
