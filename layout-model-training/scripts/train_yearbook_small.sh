#!/bin/bash

# Training script for SMALL test subset (50 images)
# Much faster for testing - will complete in 30-60 minutes

cd ../tools

python3 train_net.py \
    --dataset_name          yearbook-layout-small \
    --json_annotation_train ../data/yearbook/annotations-train-small.json \
    --image_path_train      ../data/yearbook/images \
    --json_annotation_val   ../data/yearbook/annotations-val-small.json \
    --image_path_val        ../data/yearbook/images \
    --config-file           ../configs/yearbook/fast_rcnn_R_50_FPN_3x.yaml \
    OUTPUT_DIR              ../outputs/yearbook-small/fast_rcnn_R_50_FPN_3x/ \
    SOLVER.IMS_PER_BATCH    2 \
    SOLVER.MAX_ITER         500

echo ""
echo "Training complete! Model saved to:"
echo "  outputs/yearbook-small/fast_rcnn_R_50_FPN_3x/model_final.pth"
echo ""
echo "To test your model, see TRAINING_GUIDE.md Step 5"
