# Layout Model Training for Yearbooks

Custom layout detection model training for yearbook photo detection using Detectron2.

## Overview

This directory contains tools to train a custom Faster R-CNN model to detect **human figures** (student photos, faculty photos, group photos) in yearbook pages. The model is trained on yearbook-specific layouts and filtered to exclude text boxes.

**Model Performance:**
- AP (Average Precision): 75.9%
- AP50: 97.5% (excellent detection accuracy)
- AP75: 94.7%
- Trained on 37 images with 404 human figure annotations

## Prerequisites

```bash
# Install dependencies
pip install detectron2 layoutparser torch torchvision opencv-python

# Or use requirements.txt from parent directory
cd ..
pip install -r requirements.txt
```

## Quick Start

### 1. Export Annotations from Label Studio

First, export your annotations (focusing on human figures only):

```bash
# From the repository root
python3 fetch_annotations_direct.py
```

This will:
- Fetch annotations from Label Studio project
- Filter to include only: "Student Photo", "Faculty Photo", "Group Photo"
- Exclude text labels: "Name", "Additional Text"
- Save to `layout-model-training/data/yearbook/annotations.json`

### 2. Create Training Subset

```bash
# From the repository root
python3 create_small_subset.py
```

Creates a 50-image subset (40 train, 10 validation) for quick testing.

### 3. Download Training Images

```bash
# From the repository root
python3 download_images.py
```

Downloads images from Label Studio to `layout-model-training/data/yearbook/images/`

**Note:** This directory is excluded from Git due to size (223MB). Others will need to download their own training images.

### 4. Train Model

```bash
cd layout-model-training/scripts
./train_yearbook_small.sh
```

Training takes ~1.5-2 hours on CPU (500 iterations on 37 images).

### 5. Test Trained Model

```bash
cd layout-model-training
python3 test_trained_model.py \
    --input /path/to/yearbook_page.jpg \
    --output test_results/ \
    --model outputs/yearbook-small/fast_rcnn_R_50_FPN_3x/model_final.pth \
    --config configs/yearbook/fast_rcnn_R_50_FPN_3x.yaml \
    --confidence 0.5
```

**Note:** The trained model (`model_final.pth`, 315MB) is excluded from Git. You'll need to train your own or download a pre-trained version separately.

## Directory Structure

```
layout-model-training/
├── data/yearbook/
│   ├── images/                      # Downloaded yearbook images (50 files, 232MB)
│   ├── annotations-small.json       # All annotations (50 images)
│   ├── annotations-train-small.json # Training set (40 images, 2413 photos)
│   └── annotations-val-small.json   # Validation set (10 images, 918 photos)
│
├── configs/yearbook/
│   └── fast_rcnn_R_50_FPN_3x.yaml  # Model configuration
│
├── outputs/yearbook-small/
│   └── model_final.pth             # Trained model (created after training)
│
├── scripts/
│   └── train_yearbook_small.sh     # Training script (500 iter, ~30 min)
│
└── training-small.log              # Training progress log
```

## Requirements

- Python 3.9+
- PyTorch
- Detectron2
- Label Studio API access

See main [requirements.txt](../requirements.txt) for full dependencies.

## Training Configuration

- **Model:** Faster R-CNN with ResNet-50 backbone
- **Dataset:** 50 yearbook pages, ~3,300 photo annotations
- **Split:** 40 train / 10 validation (80/20)
- **Iterations:** 500 (increase for better accuracy)
- **Batch size:** 2
- **Device:** CPU (change `MODEL.DEVICE: cuda` in config if GPU available)

## For Larger Dataset

To train on more images:

1. Modify `create_small_subset.py` to select more images (e.g., 200)
2. Run `download_images.py` to fetch additional images
3. Increase `SOLVER.MAX_ITER` in training script (e.g., 3000 for 200+ images)

Training time: ~1 hour per 1000 iterations on CPU

## Notes

- **Images must be downloaded locally** - Detectron2 reads from disk during training
- Training time scales linearly with dataset size and iterations
- Model checkpoints saved every 500 iterations to `outputs/yearbook-small/`
- GPU training is 10-20x faster than CPU
- The `.gitignore` excludes images and model files from git (they're too large)

## Original Repository

Training setup based on: https://github.com/Layout-Parser/layout-model-training

See [README_ORIGINAL.md](README_ORIGINAL.md) for original documentation.
