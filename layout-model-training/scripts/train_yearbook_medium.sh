#!/bin/bash
# Training script for medium yearbook dataset (160 images, 2 classes)
# Uses config defaults: MAX_ITER=3000, STEPS=(2000, 2500)
# Estimated time: 7-10 hours on CPU

cd /Users/louis/Desktop/yearbook-urap/layout-model-training

python3 tools/train_net.py \
    --dataset_name yearbook-layout-medium \
    --json_annotation_train data/yearbook/annotations-train-medium.json \
    --image_path_train data/yearbook/images \
    --json_annotation_val data/yearbook/annotations-val-medium.json \
    --image_path_val data/yearbook/images \
    --config-file configs/yearbook/fast_rcnn_R_50_FPN_3x.yaml \
    OUTPUT_DIR outputs/yearbook-medium-2class/fast_rcnn_R_50_FPN_3x/ \
    SOLVER.IMS_PER_BATCH 2
