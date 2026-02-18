# Model Training Setup

This directory contains tools for training custom yearbook layout detection models.

## What's Here

- **`fetch_annotations_direct.py`** - Export annotations from Label Studio
- **`create_small_subset.py`** - Create 50-image test subset for quick training
- **`download_images.py`** - Download yearbook images from Label Studio
- **`monitor_training.sh`** - Monitor training progress

See [`layout-model-training/README.md`](layout-model-training/README.md) for detailed training instructions.

## Quick Setup

```bash
# 1. Export annotations
python3 fetch_annotations_direct.py

# 2. Create test subset
python3 create_small_subset.py

# 3. Download images
python3 download_images.py

# 4. Train
cd layout-model-training/scripts && ./train_yearbook_small.sh
```

Training the small model takes ~30 minutes on CPU.
