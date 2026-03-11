#!/bin/bash

# Train text-only model after filtering suspicious giant/portrait-like text boxes.
cd "$(dirname "$0")/.."

echo "======================================"
echo "Training Cleaned Text-Only Model"
echo "======================================"
echo ""
echo "Dataset: text-only after suspicious-box filtering"
echo "Iterations: 5,000"
echo ""

python3 ../create_clean_text_training_data.py

python3 tools/train_net.py \
  --config-file configs/yearbook/fast_rcnn_R_50_FPN_text_only.yaml \
  --num-gpus 0 \
  --dataset_name yearbook_text_only_clean \
  --json_annotation_train data/yearbook/annotations-text-only-clean-train.json \
  --image_path_train data/yearbook/images \
  --json_annotation_val data/yearbook/annotations-text-only-clean-val.json \
  --image_path_val data/yearbook/images \
  2>&1 | tee scripts/train_text_only_cleaned.log
