#!/bin/bash

# Train text-only model
cd "$(dirname "$0")/.."

echo "======================================"
echo "Training Text-Only Detection Model"
echo "======================================"
echo ""
echo "Dataset: Text annotations only"
echo "Iterations: 5,000"
echo "Expected time: ~20 hours on CPU"
echo ""

# Run training
python3 tools/train_net.py \
  --config-file configs/yearbook/fast_rcnn_R_50_FPN_text_only.yaml \
  --num-gpus 0 \
  --dataset_name yearbook_text_only \
  --json_annotation_train data/yearbook/annotations-text-only-train.json \
  --image_path_train data/yearbook/images \
  --json_annotation_val data/yearbook/annotations-text-only-val.json \
  --image_path_val data/yearbook/images \
  2>&1 | tee scripts/train_text_only.log
